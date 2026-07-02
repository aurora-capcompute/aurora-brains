//go:build tinygo

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/extism/go-pdk"

	"github.com/aurora-capcompute/capcompute/sys/wire"
)

//go:wasmimport extism:host/compute syscall
func hostSyscall(uint64) uint64

const protocolPrompt = `You are an Aurora agent running inside a Wasm guest.
The host owns all side effects. Reply with exactly one compact JSON object containing an "actions" array.
Use only the tools listed below. Match each tool's input JSON schema exactly.
You may request multiple independent tool calls in one turn. The host executes them sequentially and returns one aggregated observation array.
Each observation has status "result" with content or status "failed" with an error. A failed tool call is recoverable by default: use other sources, retry when appropriate, or explain the limitation.
Add "hard": true to a tool call only when its failure must abort the run so a later resume re-executes it (for example, a state-changing step the run cannot meaningfully continue without). Omit "hard" for all normal, recoverable calls.
After receiving observations, either request more tools or return exactly one final action:
{"actions":[{"action":"final","content":{"answer":"...","reason":"..."}}]}
Never combine a final action with tool calls in the same actions array.`

type input struct {
	Message      string       `json:"message"`
	History      []message    `json:"history,omitempty"`
	SystemPrompt string       `json:"system_prompt,omitempty"`
	Capabilities []capability `json:"capabilities,omitempty"`
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type llmRequest struct {
	Messages       []message          `json:"messages"`
	ResponseFormat *llmResponseFormat `json:"response_format,omitempty"`
}

type llmResponseFormat struct {
	Type string `json:"type"`
}

type llmResponse struct {
	Choices []llmChoice `json:"choices"`
}

type llmChoice struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
}

type capability struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type modelEnvelope struct {
	Action  string          `json:"action"`
	Content json.RawMessage `json:"content"`
	// Hard marks a call whose failure must abort the run (with its savepoint left
	// open) so a later resume re-executes it, instead of being reported back as a
	// recoverable observation. Default (false) is the soft path.
	Hard bool `json:"hard,omitempty"`
}

type modelDiagnostic struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

type modelBatch struct {
	Actions json.RawMessage `json:"actions"`
}

type finalAction struct {
	Answer string `json:"answer"`
	Reason string `json:"reason,omitempty"`
}

type toolObservation struct {
	Action  string          `json:"action"`
	Status  string          `json:"status"`
	Args    json.RawMessage `json:"args,omitempty"`
	Content json.RawMessage `json:"content,omitempty"`
	Error   string          `json:"error,omitempty"`
}

type output struct {
	Status string `json:"status"`
	Answer string `json:"answer"`
}

// abiVersion is the syscall ABI this brain speaks (sys.ABIVersion in
// capcompute); the host rejects mismatches with code "bad_abi". Since v3 the
// envelope is protobuf (capcompute sys/wire — the same codec the host runs);
// args and results stay JSON payloads inside it.
const abiVersion = 3

var errYielded = errors.New("host yielded")

//go:wasmexport run
func run() int32 {
	if err := runAgent(); errors.Is(err, errYielded) {
		if outputErr := pdk.OutputJSON(output{Status: "yielded"}); outputErr != nil {
			pdk.SetError(outputErr)
			return 1
		}
		return 0
	} else if err != nil {
		pdk.SetError(err)
		return 1
	}
	return 0
}

func runAgent() error {
	in, err := fetchInput()
	if err != nil {
		return err
	}
	if in.Message == "" {
		return fmt.Errorf("message is required")
	}

	systemPrompt, err := buildSystemPrompt(in.SystemPrompt, in.Capabilities)
	if err != nil {
		return err
	}
	messages := make([]message, 0, len(in.History)+2)
	messages = append(messages, message{Role: "system", Content: systemPrompt})
	allowed := make(map[string]struct{}, len(in.Capabilities))
	for _, capability := range in.Capabilities {
		allowed[capability.Name] = struct{}{}
	}
	for i, historical := range in.History {
		if historical.Role != "user" && historical.Role != "assistant" {
			return fmt.Errorf("history message %d has unsupported role %q", i, historical.Role)
		}
		if historical.Content == "" {
			return fmt.Errorf("history message %d has empty content", i)
		}
		messages = append(messages, historical)
	}
	messages = append(messages, message{Role: "user", Content: in.Message})

	for {
		chat, err := llmChat(messages)
		if err != nil {
			return err
		}
		envelopes, err := decodeModelEnvelopes(chat)
		if err != nil {
			return fmt.Errorf("invalid model JSON: %w", err)
		}
		var firstFinal *modelEnvelope
		toolCount := 0
		for i := range envelopes {
			if envelopes[i].Action == "final" {
				if firstFinal == nil {
					firstFinal = &envelopes[i]
				}
			} else {
				toolCount++
			}
		}
		if toolCount == 0 && firstFinal != nil {
			return outputFinal(*firstFinal)
		}

		messages = append(messages, message{Role: "assistant", Content: chat})
		observations := make([]toolObservation, 0, len(envelopes))
		for i, envelope := range envelopes {
			if envelope.Action == "final" {
				// A final answer emitted alongside tool calls cannot incorporate
				// those calls' observations. Defer it and let the model answer
				// again after receiving the tool results.
				continue
			}
			if _, ok := allowed[envelope.Action]; !ok {
				return fmt.Errorf("action %d requested unavailable capability %q", i, envelope.Action)
			}
			if len(envelope.Content) == 0 {
				return fmt.Errorf("capability action %d missing content", i)
			}
			emitProgress(envelope.Action, envelope.Content)
			toolCall := wire.Syscall{Name: envelope.Action, Args: envelope.Content}
			var response wire.Response
			if envelope.Hard {
				response, err = dispatchHard(toolCall)
			} else {
				response, err = dispatch(toolCall)
			}
			if err != nil {
				return fmt.Errorf("execute capability action %d: %w", i, err)
			}
			observation := toolObservation{
				Action: envelope.Action,
				Status: statusString(response.Status),
				Args:   envelope.Content,
			}
			if response.Status == wire.StatusFailed {
				observation.Error = response.Message
			} else {
				observation.Content = response.Result
			}
			observations = append(observations, observation)
		}
		rawObservations, err := json.Marshal(observations)
		if err != nil {
			return fmt.Errorf("encode tool observations: %w", err)
		}
		messages = append(messages, message{Role: "user", Content: string(rawObservations)})
	}
}

func buildSystemPrompt(userPrompt string, capabilities []capability) (string, error) {
	var prompt strings.Builder
	if userPrompt = strings.TrimSpace(userPrompt); userPrompt != "" {
		prompt.WriteString(userPrompt)
		prompt.WriteString("\n\n")
	}
	prompt.WriteString(protocolPrompt)
	prompt.WriteString("\n\nAvailable tools for this run:")
	if len(capabilities) == 0 {
		prompt.WriteString("\nNone. Return a final action without attempting a tool call.")
		return prompt.String(), nil
	}
	for i, tool := range capabilities {
		name := strings.TrimSpace(tool.Name)
		if name == "" {
			return "", fmt.Errorf("capability %d name is required", i)
		}
		schema := tool.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{}`)
		}
		var compactSchema bytes.Buffer
		if err := json.Compact(&compactSchema, schema); err != nil {
			return "", fmt.Errorf("capability %q has invalid input schema: %w", name, err)
		}
		fmt.Fprintf(&prompt, "\n\nTool %d\nName: %s", i+1, name)
		if description := strings.TrimSpace(tool.Description); description != "" {
			fmt.Fprintf(&prompt, "\nDescription: %s", description)
		}
		fmt.Fprintf(&prompt, "\nInput JSON schema: %s", compactSchema.String())
	}
	prompt.WriteString("\n\nTool call shape:\n")
	prompt.WriteString(`{"actions":[{"action":"<exact tool name>","content":<input matching that tool's schema>}]}`)
	return prompt.String(), nil
}

func decodeModelEnvelopes(content string) ([]modelEnvelope, error) {
	return decodeModelEnvelopeStream(content, 0)
}

func decodeModelEnvelopeStream(content string, depth int) ([]modelEnvelope, error) {
	if depth > 1 {
		return nil, fmt.Errorf("nested encoded model JSON is not supported")
	}

	decoder := json.NewDecoder(strings.NewReader(content))
	var envelopes []modelEnvelope
	for {
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}

		trimmed := strings.TrimSpace(string(value))
		if trimmed == "" {
			continue
		}
		switch trimmed[0] {
		case '[':
			var batch []json.RawMessage
			if err := json.Unmarshal(value, &batch); err != nil {
				return nil, err
			}
			for _, item := range batch {
				decoded, err := decodeModelEnvelopeObject(item)
				if err != nil {
					return nil, err
				}
				envelopes = append(envelopes, decoded...)
			}
		case '{':
			decoded, err := decodeModelEnvelopeObject(value)
			if err != nil {
				return nil, err
			}
			envelopes = append(envelopes, decoded...)
		case '"':
			var encoded string
			if err := json.Unmarshal(value, &encoded); err != nil {
				return nil, err
			}
			nested, err := decodeModelEnvelopeStream(encoded, depth+1)
			if err != nil {
				return nil, err
			}
			envelopes = append(envelopes, nested...)
		default:
			return nil, fmt.Errorf("expected action object or array")
		}
	}
	if len(envelopes) == 0 {
		return nil, fmt.Errorf("model action batch is empty")
	}
	return envelopes, nil
}

func decodeModelEnvelopeObject(raw json.RawMessage) ([]modelEnvelope, error) {
	var diagnostic modelDiagnostic
	if err := json.Unmarshal(raw, &diagnostic); err != nil {
		return nil, err
	}
	if diagnostic.Error != "" {
		return nil, nil
	}

	var batch modelBatch
	if err := json.Unmarshal(raw, &batch); err != nil {
		return nil, err
	}
	if len(batch.Actions) != 0 {
		var items []json.RawMessage
		if err := json.Unmarshal(batch.Actions, &items); err != nil {
			return nil, fmt.Errorf("actions must be an array: %w", err)
		}
		if len(items) == 0 {
			return nil, fmt.Errorf("model action batch is empty")
		}
		envelopes := make([]modelEnvelope, 0, len(items))
		for _, item := range items {
			decoded, err := decodeModelEnvelopeObject(item)
			if err != nil {
				return nil, err
			}
			envelopes = append(envelopes, decoded...)
		}
		return envelopes, nil
	}

	var envelope modelEnvelope
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, err
	}
	if envelope.Action == "" {
		return nil, fmt.Errorf("action is required in model object: %s", abbreviatedJSON(raw, 300))
	}
	return []modelEnvelope{envelope}, nil
}

func abbreviatedJSON(raw json.RawMessage, limit int) string {
	value := strings.TrimSpace(string(raw))
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "[...]"
}

func outputFinal(envelope modelEnvelope) error {
	var action finalAction
	if err := decodeActionContent(envelope.Content, &action); err != nil {
		return fmt.Errorf("invalid final action: %w", err)
	}
	if action.Answer == "" {
		return fmt.Errorf("final action missing answer")
	}
	return finish(action.Answer)
}

// fetchInput retrieves the run input via the agent.input host call. Recording it
// on the journal makes replay deterministic.
func fetchInput() (input, error) {
	response, err := dispatch(wire.Syscall{Name: "agent.input"})
	if err != nil {
		return input{}, err
	}
	if response.Status != wire.StatusResult {
		return input{}, fmt.Errorf("host failure: %s", response.Message)
	}
	var in input
	if err := json.Unmarshal(response.Result, &in); err != nil {
		return input{}, fmt.Errorf("decode input: %w", err)
	}
	return in, nil
}

// finish reports the run's answer via the agent.finish host call (recorded on the
// journal, which is where the host reads the answer from) and signals completion.
func finish(answer string) error {
	args, err := json.Marshal(finishArgs{Answer: answer})
	if err != nil {
		return fmt.Errorf("encode finish: %w", err)
	}
	if _, err := dispatch(wire.Syscall{Name: "agent.finish", Args: args}); err != nil {
		return err
	}
	return pdk.OutputJSON(output{Status: "completed"})
}

type finishArgs struct {
	Answer string `json:"answer"`
}

func decodeActionContent(content json.RawMessage, target any) error {
	if len(content) == 0 {
		return fmt.Errorf("content is required")
	}
	if err := json.Unmarshal(content, target); err != nil {
		return err
	}
	return nil
}

func llmChat(messages []message) (string, error) {
	args, err := json.Marshal(llmRequest{
		Messages:       messages,
		ResponseFormat: &llmResponseFormat{Type: "json_object"},
	})
	if err != nil {
		return "", fmt.Errorf("encode llm request: %w", err)
	}
	response, err := dispatch(wire.Syscall{Name: "openai.chat", Args: args})
	if err != nil {
		return "", err
	}
	if response.Status != wire.StatusResult {
		return "", fmt.Errorf("host failure: %s", response.Message)
	}
	var chat llmResponse
	if err := json.Unmarshal(response.Result, &chat); err != nil {
		return "", fmt.Errorf("decode llm response: %w", err)
	}
	if len(chat.Choices) == 0 {
		return "", fmt.Errorf("provider returned no choices")
	}
	return chat.Choices[0].Message.Content, nil
}

func emitProgress(action string, content json.RawMessage) {
	summary := progressSummary(action, content)
	msg, _ := json.Marshal(map[string]string{"message": summary})
	dispatch(wire.Syscall{Name: "aurora.log", Args: msg})
}

func progressSummary(action string, content json.RawMessage) string {
	var fields map[string]json.RawMessage
	if json.Unmarshal(content, &fields) != nil {
		return "⚙ " + action
	}
	// A delegation to a sub-agent carries a free-text message; tools are now
	// addressed by their local name, so branch on the content shape rather than
	// on a name prefix.
	if msg, ok := fields["message"]; ok {
		var s string
		if json.Unmarshal(msg, &s) == nil && len(s) > 0 {
			if len(s) > 80 {
				s = s[:80] + "…"
			}
			return "🔀 " + action + ": " + s
		}
		return "🔀 " + action
	}
	// Otherwise surface identifying fields common to operational tools (the verb
	// discriminator plus resource coordinates).
	var parts []string
	for _, key := range []string{"verb", "kind", "namespace", "name", "release", "chart", "api_version"} {
		if raw, ok := fields[key]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil && s != "" {
				parts = append(parts, s)
			}
		}
	}
	if len(parts) > 0 {
		return "⚙ " + action + " " + strings.Join(parts, "/")
	}
	return "⚙ " + action
}

func dispatch(c wire.Syscall) (wire.Response, error) {
	c.Abi = abiVersion
	request := pdk.AllocateBytes(wire.EncodeSyscall(c))
	defer request.Free()

	responseOffset := hostSyscall(request.Offset())
	response, err := wire.DecodeResponse(pdk.ParamBytes(responseOffset))
	if err != nil {
		return wire.Response{}, fmt.Errorf("decode host response: %w", err)
	}
	switch response.Status {
	case wire.StatusResult, wire.StatusFailed:
		return response, nil
	case wire.StatusYield:
		return wire.Response{}, errYielded
	default:
		return wire.Response{}, fmt.Errorf("unsupported host outcome: %d", response.Status)
	}
}

// statusString renders a wire status for the LLM-facing observation JSON.
func statusString(status wire.Status) string {
	switch status {
	case wire.StatusResult:
		return "result"
	case wire.StatusYield:
		return "yield"
	case wire.StatusFailed:
		return "failed"
	default:
		return "unspecified"
	}
}

// Reserved savepoint syscalls (sys.SyscallBegin/sys.SyscallCommit in
// capcompute). They carry no side effect; the host records them on the journal
// and uses an open sys.begin (one with no matching sys.commit) as the fork
// point when a failed run is resumed. Brackets have stack semantics.
const (
	capTry    = "sys.begin"
	capCommit = "sys.commit"
)

// dispatchHard brackets a single call in a host.try/host.commit savepoint. On
// success it commits and returns the result. On failure it leaves the try open
// and returns an error that aborts the run, so a later resume forks right after
// the try and re-executes the call under a new revision. A plain dispatch (the
// default, "soft") instead records the failure for replay and lets the brain
// react to it.
func dispatchHard(c wire.Syscall) (wire.Response, error) {
	if _, err := dispatch(wire.Syscall{Name: capTry}); err != nil {
		return wire.Response{}, err
	}
	response, err := dispatch(c)
	if err != nil {
		return wire.Response{}, err
	}
	if response.Status == wire.StatusFailed {
		return response, fmt.Errorf("hard capability %q failed: %s", c.Name, response.Message)
	}
	if _, err := dispatch(wire.Syscall{Name: capCommit}); err != nil {
		return wire.Response{}, err
	}
	return response, nil
}

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
)

//go:wasmimport extism:host/compute play
func hostPlay(uint64) uint64

const protocolPrompt = `You are an Aurora agent running inside a Wasm guest.
The host owns all side effects. Reply with exactly one compact JSON object containing an "actions" array.
Use only the tools listed below. Match each tool's input JSON schema exactly.
You may request multiple independent tool calls in one turn. The host executes them sequentially and returns one aggregated observation array.
Each observation has status "result" with content or status "failed" with an error. A failed tool call is recoverable: use other sources, retry when appropriate, or explain the limitation.
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
	Messages []message `json:"messages"`
	JSON     bool      `json:"json"`
}

type llmResponse struct {
	Content string `json:"content"`
}

type capability struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type modelEnvelope struct {
	Action  string          `json:"action"`
	Content json.RawMessage `json:"content"`
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

type call struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type hostResponse struct {
	Status  string          `json:"status"`
	Result  json.RawMessage `json:"result,omitempty"`
	Message string          `json:"message,omitempty"`
}

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
	var in input
	if err := pdk.InputJSON(&in); err != nil {
		return fmt.Errorf("decode input: %w", err)
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
		envelopes, err := decodeModelEnvelopes(chat.Content)
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

		messages = append(messages, message{Role: "assistant", Content: chat.Content})
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
			response, err := dispatch(call{Name: envelope.Action, Args: envelope.Content})
			if err != nil {
				return fmt.Errorf("execute capability action %d: %w", i, err)
			}
			observation := toolObservation{
				Action: envelope.Action,
				Status: response.Status,
				Args:   envelope.Content,
			}
			if response.Status == "failed" {
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
		messages = append(messages, message{Role: "tool", Content: string(rawObservations)})
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
	return pdk.OutputJSON(output{
		Status: "completed",
		Answer: action.Answer,
	})
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

func llmChat(messages []message) (llmResponse, error) {
	args, err := json.Marshal(llmRequest{Messages: messages, JSON: true})
	if err != nil {
		return llmResponse{}, fmt.Errorf("encode llm request: %w", err)
	}
	response, err := dispatch(call{Name: "llm.chat", Args: args})
	if err != nil {
		return llmResponse{}, err
	}
	if response.Status != "result" {
		return llmResponse{}, fmt.Errorf("host failure: %s", response.Message)
	}
	var chat llmResponse
	if err := json.Unmarshal(response.Result, &chat); err != nil {
		return llmResponse{}, fmt.Errorf("decode llm response: %w", err)
	}
	return chat, nil
}

func dispatch(c call) (hostResponse, error) {
	data, err := json.Marshal(c)
	if err != nil {
		return hostResponse{}, fmt.Errorf("encode call: %w", err)
	}

	request := pdk.AllocateBytes(data)
	defer request.Free()

	responseOffset := hostPlay(request.Offset())
	var response hostResponse
	if err := pdk.JSONFrom(responseOffset, &response); err != nil {
		return hostResponse{}, fmt.Errorf("decode host response: %w", err)
	}
	switch response.Status {
	case "result", "failed":
		return response, nil
	case "yield":
		return hostResponse{}, errYielded
	default:
		return hostResponse{}, fmt.Errorf("unsupported host outcome: %s", response.Status)
	}
}

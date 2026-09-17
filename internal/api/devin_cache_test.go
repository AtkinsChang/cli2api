package api

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"sync"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/caigee-cmd/cli2api/internal/accounts"
	"github.com/caigee-cmd/cli2api/internal/auth"
	"github.com/caigee-cmd/cli2api/internal/executor"
	"github.com/caigee-cmd/cli2api/internal/providers"
	"github.com/caigee-cmd/cli2api/internal/providers/devin"
)

type devinCacheStore struct {
	account accounts.Account
	cred    []byte
}

func (s *devinCacheStore) Get(_ context.Context, id string) (accounts.Account, error) {
	if id != s.account.ID {
		return accounts.Account{}, accounts.ErrAccountNotFound
	}
	return s.account, nil
}

func (s *devinCacheStore) LoadCredentialPayload(_ context.Context, id string) (string, []byte, error) {
	if id != s.account.ID {
		return "", nil, accounts.ErrAccountNotFound
	}
	return devin.CredentialFormat, append([]byte(nil), s.cred...), nil
}

func (s *devinCacheStore) SaveCredentialPayload(_ context.Context, _ string, _ string, payload []byte) error {
	s.cred = append([]byte(nil), payload...)
	return nil
}

func (s *devinCacheStore) Observe(context.Context, string, string, string, string, string) error {
	return nil
}

type devinCacheWireRequest struct {
	session   string
	cascade   string
	execution string
	steps     uint64
	stepType  uint64
	systemTTL bool
	lastTTL   bool
}

func TestAPIChatCarriesScopedSessionToDevinCacheFields(t *testing.T) {
	var (
		mu       sync.Mutex
		captured []devinCacheWireRequest
	)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream body: %v", err)
			return
		}
		wire, err := decodeDevinCacheRequest(body)
		if err != nil {
			t.Errorf("decode upstream request: %v", err)
			return
		}
		mu.Lock()
		captured = append(captured, wire)
		mu.Unlock()
		w.Header().Set("Content-Type", devin.ContentTypeConnectProto)
		_, _ = w.Write(devin.WrapConnectEnvelopeWithFlag(devin.ConnectFlagEndStream, []byte(`{}`)))
	}))
	defer upstream.Close()

	store := &devinCacheStore{account: accounts.Account{ID: "devin-a", Provider: "devin", ProviderRegion: "global"}}
	credential, err := (devin.Credential{
		SessionToken: devin.FormatSessionToken("eyJabc.def.ghi"),
		DeviceSeed:   "test-device-seed",
		BaseURL:      upstream.URL,
	}).Encode()
	if err != nil {
		t.Fatal(err)
	}
	store.cred = credential
	client := devin.NewClient(store)
	registry := providers.NewRegistry()
	registry.Register(client.Adapter())
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "devin-a", Provider: "devin", Region: "global", Runtime: string(providers.RuntimeInProcess)})
	chatExecutor := executor.NewChatExecutor(pool, "")
	chatExecutor.Providers = registry
	server := &Server{executor: chatExecutor, pool: pool}

	first := `{"model":"devin/swe-2","messages":[{"role":"system","content":"Follow the repository instructions."},{"role":"user","content":"Inspect the failing test."}]}`
	toolContinuation := `{"model":"devin/swe-2","messages":[{"role":"system","content":"Follow the repository instructions."},{"role":"user","content":"Inspect the failing test."},{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"read_file","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call-1","content":"test contents"}]}`
	laterStream := `{"model":"devin/swe-2","stream":true,"messages":[{"role":"system","content":"Follow the repository instructions."},{"role":"user","content":"Inspect the failing test."},{"role":"assistant","content":"I found it."},{"role":"user","content":"Apply the narrow fix."}]}`
	contentFirst := `{"model":"devin/swe-2","messages":[{"role":"user","content":"Find the regression."}]}`
	contentLater := `{"model":"devin/swe-2","messages":[{"role":"user","content":"Find the regression."},{"role":"assistant","content":"Located."},{"role":"user","content":"Explain it."}]}`

	post := func(body, session, identity string) {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
		if session != "" {
			req.Header.Set("X-CLI2API-Session", session)
		}
		req = req.WithContext(auth.WithIdentity(req.Context(), auth.Identity{Kind: auth.KindKey, KeyID: identity}))
		recorder := httptest.NewRecorder()
		server.handleChatCompletions(recorder, req)
		if recorder.Code != http.StatusOK {
			t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
		}
	}

	post(first, "conversation-1", "key-a")
	post(first, "conversation-1", "key-a")
	post(toolContinuation, "conversation-1", "key-a")
	post(toolContinuation, "conversation-1", "key-a")
	post(laterStream, "conversation-1", "key-a")
	post(first, "conversation-1", "key-b")
	post(contentFirst, "", "key-a")
	post(contentLater, "", "key-a")
	post(first, "conversation-1", "key-a")
	post(`{"model":"devin/swe-2","messages":[{"role":"system","content":"Follow the repository instructions."},{"role":"user","content":"Inspect the failing test."},{"role":"assistant","content":"I found it."}]}`, "conversation-1", "key-a")

	mu.Lock()
	got := append([]devinCacheWireRequest(nil), captured...)
	mu.Unlock()
	if len(got) != 10 {
		t.Fatalf("captured %d requests, want 10", len(got))
	}
	uuidV4 := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)
	for _, request := range got {
		for _, id := range []string{request.session, request.cascade, request.execution} {
			if !uuidV4.MatchString(id) {
				t.Fatalf("not a UUID v4: %q", id)
			}
		}
		if request.session == request.cascade || request.execution == request.session || request.execution == request.cascade {
			t.Fatalf("IDs are not independent: %+v", request)
		}
	}
	for i := 1; i <= 4; i++ {
		if got[0].session != got[i].session || got[0].cascade != got[i].cascade {
			t.Fatalf("same scoped session changed at %d: first=%+v later=%+v", i, got[0], got[i])
		}
	}
	for i := 1; i <= 3; i++ {
		if got[0].execution != got[i].execution {
			t.Fatalf("retry/tool continuation changed execution at %d: %+v", i, got)
		}
	}
	if got[0].execution == got[4].execution {
		t.Fatalf("new user turn reused execution: %+v", got[4])
	}
	for _, i := range []int{8, 9} {
		if got[i].session != got[0].session || got[i].cascade != got[0].cascade || got[i].execution != got[0].execution {
			t.Fatalf("earlier turn lost its identity at %d: %+v", i, got)
		}
	}
	if got[8].steps != 1 || got[9].steps != 2 || got[9].stepType != 15 {
		t.Fatalf("replayed history did not determine trajectory reference: %+v", got[8:])
	}
	if got[0].session == got[5].session || got[0].cascade == got[5].cascade {
		t.Fatalf("identity scope did not separate IDs: key-a=%+v key-b=%+v", got[0], got[5])
	}
	if got[6].session != got[7].session || got[6].cascade != got[7].cascade {
		t.Fatalf("content fallback was not stable: first=%+v later=%+v", got[6], got[7])
	}
	for i, steps := range []uint64{1, 1, 3, 3, 3} {
		if got[i].steps != steps {
			t.Fatalf("request %d step index=%d, want history length %d", i, got[i].steps, steps)
		}
	}
	if got[0].stepType != 14 || got[4].stepType != 14 || got[2].stepType != 0 {
		t.Fatalf("last prompt step types do not reflect user/unknown-tool boundaries: %+v", got)
	}
	if !got[4].systemTTL || !got[4].lastTTL {
		t.Fatalf("missing EPHEMERAL cache markers: %+v", got[4])
	}
}

func decodeDevinCacheRequest(body []byte) (devinCacheWireRequest, error) {
	_, payload, err := devin.ReadConnectFrame(bytes.NewReader(body))
	if err != nil {
		return devinCacheWireRequest{}, err
	}
	fields, err := devinCacheFields(payload)
	if err != nil {
		return devinCacheWireRequest{}, err
	}
	var result devinCacheWireRequest
	if session := fields[15]; len(session) == 1 {
		sessionFields, err := devinCacheFields(session[0])
		if err != nil {
			return result, err
		}
		if values := sessionFields[1]; len(values) == 1 {
			result.session = string(values[0])
		}
		result.steps, err = devinCacheVarint(session[0], 2)
		if err != nil {
			return result, err
		}
		result.stepType, err = devinCacheVarint(session[0], 4)
		if err != nil {
			return result, err
		}
	}
	if values := fields[16]; len(values) == 1 {
		result.cascade = string(values[0])
	}
	if values := fields[22]; len(values) == 1 {
		result.execution = string(values[0])
	}
	result.systemTTL = devinCacheEphemeral(fields[13])
	if prompts := fields[3]; len(prompts) > 0 {
		promptFields, err := devinCacheFields(prompts[len(prompts)-1])
		if err != nil {
			return result, err
		}
		result.lastTTL = devinCacheEphemeral(promptFields[8])
	}
	return result, nil
}

func devinCacheVarint(data []byte, wanted protowire.Number) (uint64, error) {
	for len(data) > 0 {
		number, kind, n := protowire.ConsumeTag(data)
		if n < 0 {
			return 0, protowire.ParseError(n)
		}
		data = data[n:]
		if number == wanted && kind == protowire.VarintType {
			value, used := protowire.ConsumeVarint(data)
			if used < 0 {
				return 0, protowire.ParseError(used)
			}
			return value, nil
		}
		n = protowire.ConsumeFieldValue(number, kind, data)
		if n < 0 {
			return 0, protowire.ParseError(n)
		}
		data = data[n:]
	}
	return 0, nil
}

func devinCacheEphemeral(values [][]byte) bool {
	return len(values) == 1 && bytes.Equal(values[0], []byte{0x08, 0x01})
}

func devinCacheFields(data []byte) (map[int][][]byte, error) {
	fields := map[int][][]byte{}
	for len(data) > 0 {
		number, typ, size := protowire.ConsumeTag(data)
		if size <= 0 {
			return nil, fmt.Errorf("invalid protobuf tag")
		}
		data = data[size:]
		if typ == protowire.BytesType {
			value, used := protowire.ConsumeBytes(data)
			if used <= 0 {
				return nil, fmt.Errorf("invalid bytes field %d", number)
			}
			fields[int(number)] = append(fields[int(number)], append([]byte(nil), value...))
			data = data[used:]
			continue
		}
		used := protowire.ConsumeFieldValue(number, typ, data)
		if used <= 0 {
			return nil, fmt.Errorf("invalid field %d", number)
		}
		data = data[used:]
	}
	return fields, nil
}

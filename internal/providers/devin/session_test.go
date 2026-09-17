package devin

import (
	"sync"
	"testing"
	"time"

	cortexpb "github.com/caigee-cmd/cli2api/internal/providers/devin/devinpb/cortex_pb"
)

func identityPayload(prompts ...Prompt) ChatPayload {
	return ChatPayload{
		ModelUID: "swe-2", System: "follow instructions", Tools: []Tool{{Name: "read_file", Parameters: []byte(`{"type":"object"}`)}},
		MaxTokens: 4096, Prompts: prompts,
	}
}

func TestChatIdentityRetriesToolContinuationsAndNewUsers(t *testing.T) {
	client := NewClient(nil)
	first := identityPayload(Prompt{Source: 1, Content: "inspect"})
	retry, err := client.chatIdentity("account-a", "device-a", "session-a", first)
	if err != nil {
		t.Fatal(err)
	}
	again, err := client.chatIdentity("account-a", "device-a", "session-a", first)
	if err != nil {
		t.Fatal(err)
	}
	tool := identityPayload(
		Prompt{Source: 1, Content: "inspect"},
		Prompt{Source: 2, Content: "", ToolCalls: []ToolCall{{ID: "call", Name: "read_file", Arguments: `{}`}}},
		Prompt{Source: 4, Content: "contents", ToolCallID: "call"},
	)
	continued, err := client.chatIdentity("account-a", "device-a", "session-a", tool)
	if err != nil {
		t.Fatal(err)
	}
	later := identityPayload(
		Prompt{Source: 1, Content: "inspect"}, Prompt{Source: 2, Content: "found it"}, Prompt{Source: 1, Content: "fix it"},
	)
	next, err := client.chatIdentity("account-a", "device-a", "session-a", later)
	if err != nil {
		t.Fatal(err)
	}
	for _, got := range []ChatIdentity{again, continued, next} {
		if got.TrajectoryID != retry.TrajectoryID || got.CascadeID != retry.CascadeID {
			t.Fatalf("scoped IDs changed: first=%+v later=%+v", retry, got)
		}
	}
	if again.ExecutionID != retry.ExecutionID || continued.ExecutionID != retry.ExecutionID {
		t.Fatalf("retry/tool continuation changed execution: retry=%+v continued=%+v", retry, continued)
	}
	if next.ExecutionID == retry.ExecutionID {
		t.Fatalf("new user turn reused execution: first=%+v next=%+v", retry, next)
	}
	if retry.StepIndex != 1 || retry.StepType != cortexpb.CortexStepType_CORTEX_STEP_TYPE_USER_INPUT {
		t.Fatalf("first trajectory metadata = %+v", retry)
	}
	if continued.StepIndex != 3 || continued.StepType != cortexpb.CortexStepType_CORTEX_STEP_TYPE_UNSPECIFIED {
		t.Fatalf("tool trajectory metadata = %+v", continued)
	}
}

func TestChatIdentitySeparatesAccountsDevicesAndConfiguration(t *testing.T) {
	client := NewClient(nil)
	payload := identityPayload(Prompt{Source: 1, Content: "inspect"})
	base, err := client.chatIdentity("account-a", "device-a", "session-a", payload)
	if err != nil {
		t.Fatal(err)
	}
	otherAccount, err := client.chatIdentity("account-b", "device-a", "session-a", payload)
	if err != nil {
		t.Fatal(err)
	}
	otherDevice, err := client.chatIdentity("account-a", "device-b", "session-a", payload)
	if err != nil {
		t.Fatal(err)
	}
	changed := payload
	changed.System = "different instructions"
	configChanged, err := client.chatIdentity("account-a", "device-a", "session-a", changed)
	if err != nil {
		t.Fatal(err)
	}
	if base.TrajectoryID == otherAccount.TrajectoryID || base.CascadeID == otherAccount.CascadeID || base.TrajectoryID == otherDevice.TrajectoryID || base.CascadeID == otherDevice.CascadeID {
		t.Fatal("account/device scope reused trajectory or cascade")
	}
	if base.ExecutionID == configChanged.ExecutionID {
		t.Fatal("changed configuration reused execution")
	}
	isolatedA, err := client.chatIdentity("account-a", "device-a", "", payload)
	if err != nil {
		t.Fatal(err)
	}
	isolatedB, err := client.chatIdentity("account-a", "device-a", "", payload)
	if err != nil {
		t.Fatal(err)
	}
	if isolatedA == isolatedB {
		t.Fatal("empty scope reused identity")
	}
}

func TestChatIdentityCacheExpiresEvictsAndBoundsExecutions(t *testing.T) {
	cache := newChatIdentityCache(time.Hour, 2)
	firstTrajectory, firstCascade, firstExecution, err := cache.identity("one", "first")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := cache.identity("two", "first"); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := cache.identity("one", "first"); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := cache.identity("three", "first"); err != nil {
		t.Fatal(err)
	}
	cache.mu.Lock()
	_, twoPresent := cache.entries["two"]
	cache.mu.Unlock()
	if twoPresent {
		t.Fatal("least recently used session was not evicted")
	}
	for index := 0; index <= chatExecutionCapacity; index++ {
		if _, _, _, err := cache.identity("one", string(rune(index))); err != nil {
			t.Fatal(err)
		}
	}
	cache.mu.Lock()
	entry := cache.entries["one"].Value.(chatIdentityEntry)
	cache.mu.Unlock()
	if entry.executionLRU.Len() != chatExecutionCapacity {
		t.Fatalf("execution entries = %d, want %d", entry.executionLRU.Len(), chatExecutionCapacity)
	}
	priorTrajectory, priorCascade, priorExecution, err := cache.identity("one", "first")
	if err != nil {
		t.Fatal(err)
	}
	if priorTrajectory != firstTrajectory || priorCascade != firstCascade || priorExecution == firstExecution {
		t.Fatalf("execution LRU or session identity changed unexpectedly: first=%q/%q/%q prior=%q/%q/%q", firstTrajectory, firstCascade, firstExecution, priorTrajectory, priorCascade, priorExecution)
	}
	cache.mu.Lock()
	element := cache.entries["one"]
	entry = element.Value.(chatIdentityEntry)
	oldExpiry := time.Now().Add(time.Minute)
	entry.expiresAt = oldExpiry
	element.Value = entry
	cache.mu.Unlock()
	if _, _, _, err := cache.identity("one", "refresh"); err != nil {
		t.Fatal(err)
	}
	cache.mu.Lock()
	refreshed := cache.entries["one"].Value.(chatIdentityEntry).expiresAt
	entry = cache.entries["one"].Value.(chatIdentityEntry)
	entry.expiresAt = time.Now().Add(-time.Second)
	element.Value = entry
	cache.mu.Unlock()
	if !refreshed.After(oldExpiry) {
		t.Fatalf("cache hit did not refresh idle expiry: old=%v new=%v", oldExpiry, refreshed)
	}
	afterTrajectory, afterCascade, afterExecution, err := cache.identity("one", "first")
	if err != nil {
		t.Fatal(err)
	}
	if afterTrajectory == priorTrajectory || afterCascade == priorCascade || afterExecution == priorExecution {
		t.Fatalf("expired session reused identity: prior=%q/%q/%q after=%q/%q/%q", priorTrajectory, priorCascade, priorExecution, afterTrajectory, afterCascade, afterExecution)
	}
}

func TestChatIdentityConcurrentSameKey(t *testing.T) {
	client := NewClient(nil)
	payload := identityPayload(Prompt{Source: 1, Content: "inspect"})
	identities := make(chan ChatIdentity, 32)
	errs := make(chan error, 32)
	var group sync.WaitGroup
	for range 32 {
		group.Add(1)
		go func() {
			defer group.Done()
			identity, err := client.chatIdentity("account-a", "device-a", "session-a", payload)
			if err != nil {
				errs <- err
				return
			}
			identities <- identity
		}()
	}
	group.Wait()
	close(errs)
	close(identities)
	for err := range errs {
		t.Fatal(err)
	}
	var first ChatIdentity
	for identity := range identities {
		if first.TrajectoryID == "" {
			first = identity
			continue
		}
		if identity != first {
			t.Fatalf("concurrent allocation disagreed: first=%+v got=%+v", first, identity)
		}
	}
}

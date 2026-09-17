package devin

import (
	"container/list"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	cortexpb "github.com/caigee-cmd/cli2api/internal/providers/devin/devinpb/cortex_pb"
)

const (
	chatIdentityTTL       = time.Hour
	chatIdentityCapacity  = 10_000
	chatExecutionCapacity = 64
)

// ChatIdentity is the request-scoped subset of Desktop-like identifiers.
type ChatIdentity struct {
	TrajectoryID string
	CascadeID    string
	ExecutionID  string
	StepIndex    int32
	StepType     cortexpb.CortexStepType
}

type chatIdentityEntry struct {
	key          string
	trajectoryID string
	cascadeID    string
	expiresAt    time.Time
	executions   map[string]*list.Element
	executionLRU *list.List
}

type chatExecutionEntry struct {
	key string
	id  string
}

// LIMIT: This normalized projection is not Desktop's private step model. IDs
// are process-local, with a one-hour idle TTL, 10k-session LRU, and 64 replay
// executions per session; revisit if upstream publishes durable identities.
type chatIdentityCache struct {
	mu       sync.Mutex
	ttl      time.Duration
	capacity int
	entries  map[string]*list.Element
	lru      *list.List
}

func newChatIdentityCache(ttl time.Duration, capacity int) *chatIdentityCache {
	if ttl <= 0 {
		ttl = chatIdentityTTL
	}
	if capacity <= 0 {
		capacity = chatIdentityCapacity
	}
	return &chatIdentityCache{
		ttl:      ttl,
		capacity: capacity,
		entries:  make(map[string]*list.Element, capacity),
		lru:      list.New(),
	}
}

func (c *chatIdentityCache) identity(scope, executionKey string) (trajectoryID, cascadeID, executionID string, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	element, ok := c.entries[scope]
	if ok {
		entry := element.Value.(chatIdentityEntry)
		if entry.expiresAt.After(time.Now()) {
			c.lru.MoveToFront(element)
			entry.expiresAt = time.Now().Add(c.ttl)
			element.Value = entry
			return c.executionFor(&entry, element, executionKey)
		}
		c.remove(element)
	}
	trajectoryID, err = randomUUIDv4()
	if err != nil {
		return "", "", "", err
	}
	cascadeID, err = randomUUIDv4()
	if err != nil {
		return "", "", "", err
	}
	executionID, err = randomUUIDv4()
	if err != nil {
		return "", "", "", err
	}
	entry := chatIdentityEntry{
		key: scope, trajectoryID: trajectoryID, cascadeID: cascadeID,
		expiresAt: time.Now().Add(c.ttl), executions: make(map[string]*list.Element), executionLRU: list.New(),
	}
	if executionKey != "" {
		entry.addExecution(executionKey, executionID)
	}
	element = c.lru.PushFront(entry)
	c.entries[scope] = element
	for c.lru.Len() > c.capacity {
		c.remove(c.lru.Back())
	}
	return trajectoryID, cascadeID, executionID, nil
}

func (c *chatIdentityCache) executionFor(entry *chatIdentityEntry, element *list.Element, executionKey string) (string, string, string, error) {
	if executionKey == "" {
		executionID, err := randomUUIDv4()
		return entry.trajectoryID, entry.cascadeID, executionID, err
	}
	if execution, ok := entry.executions[executionKey]; ok {
		c := execution.Value.(chatExecutionEntry)
		entry.executionLRU.MoveToFront(execution)
		return entry.trajectoryID, entry.cascadeID, c.id, nil
	}
	executionID, err := randomUUIDv4()
	if err != nil {
		return "", "", "", err
	}
	entry.addExecution(executionKey, executionID)
	element.Value = *entry
	return entry.trajectoryID, entry.cascadeID, executionID, nil
}

func (entry *chatIdentityEntry) addExecution(key, id string) {
	element := entry.executionLRU.PushFront(chatExecutionEntry{key: key, id: id})
	entry.executions[key] = element
	for entry.executionLRU.Len() > chatExecutionCapacity {
		oldest := entry.executionLRU.Back()
		delete(entry.executions, oldest.Value.(chatExecutionEntry).key)
		entry.executionLRU.Remove(oldest)
	}
}

func (c *chatIdentityCache) remove(element *list.Element) {
	if element == nil {
		return
	}
	entry := element.Value.(chatIdentityEntry)
	delete(c.entries, entry.key)
	c.lru.Remove(element)
}

func (c *Client) chatIdentityCache() *chatIdentityCache {
	c.identityMu.Lock()
	defer c.identityMu.Unlock()
	if c.identities == nil {
		c.identities = newChatIdentityCache(chatIdentityTTL, chatIdentityCapacity)
	}
	return c.identities
}

func (c *Client) chatIdentity(accountID, deviceSeed, sessionKey string, payload ChatPayload) (ChatIdentity, error) {
	identity := ChatIdentity{StepIndex: int32(len(payload.Prompts)), StepType: chatStepType(payload.Prompts)}
	scope := chatIdentityScope(accountID, deviceSeed, sessionKey)
	if scope == "" {
		return allocateIsolatedChatIdentity(identity)
	}
	executionKey, err := chatExecutionKey(payload)
	if err != nil {
		return ChatIdentity{}, fmt.Errorf("hash execution identity: %w", err)
	}
	cache := c.chatIdentityCache()
	identity.TrajectoryID, identity.CascadeID, identity.ExecutionID, err = cache.identity(scope, executionKey)
	if err != nil {
		return ChatIdentity{}, fmt.Errorf("allocate chat identity: %w", err)
	}
	return identity, nil
}

func allocateIsolatedChatIdentity(identity ChatIdentity) (ChatIdentity, error) {
	var err error
	if identity.TrajectoryID, err = randomUUIDv4(); err != nil {
		return ChatIdentity{}, fmt.Errorf("allocate trajectory ID: %w", err)
	}
	if identity.CascadeID, err = randomUUIDv4(); err != nil {
		return ChatIdentity{}, fmt.Errorf("allocate cascade ID: %w", err)
	}
	if identity.ExecutionID, err = randomUUIDv4(); err != nil {
		return ChatIdentity{}, fmt.Errorf("allocate execution ID: %w", err)
	}
	return identity, nil
}

func chatIdentityScope(accountID, deviceSeed, sessionKey string) string {
	accountID = strings.TrimSpace(accountID)
	deviceSeed = strings.TrimSpace(deviceSeed)
	sessionKey = strings.TrimSpace(sessionKey)
	if accountID == "" || deviceSeed == "" || sessionKey == "" {
		return ""
	}
	return chatIdentityDigest("scope", accountID, deviceSeed, sessionKey)
}

type executionIdentityInput struct {
	ModelUID    string
	System      string
	Tools       []Tool
	Temperature *float64
	MaxTokens   int
	Prompts     []Prompt
}

func chatExecutionKey(payload ChatPayload) (string, error) {
	lastUser := -1
	for index, prompt := range payload.Prompts {
		if prompt.Source == 1 {
			lastUser = index
		}
	}
	if lastUser < 0 {
		return "", nil
	}
	input := executionIdentityInput{
		ModelUID: payload.ModelUID, System: payload.System, Tools: payload.Tools,
		Temperature: payload.Temperature, MaxTokens: payload.MaxTokens,
		Prompts: payload.Prompts[:lastUser+1],
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		return "", err
	}
	return chatIdentityDigest("execution", string(encoded)), nil
}

func chatIdentityDigest(kind string, values ...string) string {
	hash := sha256.New()
	hash.Write([]byte(kind))
	var length [8]byte
	for _, value := range values {
		binary.BigEndian.PutUint64(length[:], uint64(len(value)))
		hash.Write(length[:])
		hash.Write([]byte(value))
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func chatStepType(prompts []Prompt) cortexpb.CortexStepType {
	if len(prompts) == 0 {
		return cortexpb.CortexStepType_CORTEX_STEP_TYPE_UNSPECIFIED
	}
	switch prompts[len(prompts)-1].Source {
	case 1:
		return cortexpb.CortexStepType_CORTEX_STEP_TYPE_USER_INPUT
	case 2:
		return cortexpb.CortexStepType_CORTEX_STEP_TYPE_PLANNER_RESPONSE
	case 4:
		// The generated protocol has no truthful generic TOOL_RESULT step type.
		return cortexpb.CortexStepType_CORTEX_STEP_TYPE_UNSPECIFIED
	default:
		return cortexpb.CortexStepType_CORTEX_STEP_TYPE_UNSPECIFIED
	}
}

func randomUUIDv4() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", fmt.Errorf("read UUID randomness: %w", err)
	}
	value[6] = value[6]&0x0f | 0x40
	value[8] = value[8]&0x3f | 0x80
	encoded := hex.EncodeToString(value[:])
	return encoded[0:8] + "-" + encoded[8:12] + "-" + encoded[12:16] + "-" + encoded[16:20] + "-" + encoded[20:], nil
}

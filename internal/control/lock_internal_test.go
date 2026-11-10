//go:build unix

package control

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestLockInstancesSerializes(t *testing.T) {
	cfg := instanceTestConfig(t, "")
	var inside, overlaps atomic.Int32
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			unlock, err := LockInstances(cfg)
			if err != nil {
				t.Error(err)
				return
			}
			if inside.Add(1) > 1 {
				overlaps.Add(1)
			}
			time.Sleep(20 * time.Millisecond)
			inside.Add(-1)
			unlock()
		}()
	}
	wg.Wait()
	if n := overlaps.Load(); n != 0 {
		t.Fatalf("%d callers held the lock at once", n)
	}
	insts, err := ListInstances(cfg)
	if err != nil || len(insts) != 0 {
		t.Fatalf("lock file listed as an instance: %v %v", insts, err)
	}
}

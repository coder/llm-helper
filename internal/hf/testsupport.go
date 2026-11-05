package hf

// SetBaseURLForTest points the Hub client at a test server and returns a
// function that restores the previous value.
//
// Exported because resolve's tests need it as well: the judgements worth
// testing -- which quants are offered, what is refused -- live there, and
// they need a Hub that answers predictably. It is inert outside a test, since
// nothing else ever calls it.
func SetBaseURLForTest(u string) func() {
	prev := baseURL
	baseURL = u
	return func() { baseURL = prev }
}

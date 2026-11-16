//go:build !linux && !darwin

package hfcache

import "errors"

// freeBytes is unknown here, which CheckSpace reads as nothing to check.
var freeBytes = func(string) (int64, error) { return 0, errors.New("free space is not known on this platform") }

// onFAT is false where the filesystem cannot be read.
func onFAT(string) bool { return false }

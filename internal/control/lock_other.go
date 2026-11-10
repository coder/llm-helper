//go:build !unix

package control

import "os"

func lockFile(*os.File) error   { return nil }
func unlockFile(*os.File) error { return nil }

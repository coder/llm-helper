package hfcache

import "syscall"

// freeBytes is what a user can still write to the filesystem holding dir. It
// is a variable so tests can fix it.
var freeBytes = func(dir string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), nil
}

// onFAT reports whether dir is on a FAT filesystem, which macOS calls msdos.
func onFAT(dir string) bool {
	var st syscall.Statfs_t
	if syscall.Statfs(dir, &st) != nil {
		return false
	}
	var name []byte
	for _, c := range st.Fstypename {
		if c == 0 {
			break
		}
		name = append(name, byte(c))
	}
	return string(name) == "msdos"
}

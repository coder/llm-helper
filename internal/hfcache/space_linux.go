package hfcache

import "syscall"

// msdosSuperMagic is the filesystem type of FAT12, FAT16, FAT32 and VFAT.
const msdosSuperMagic = 0x4d44

// freeBytes is what a user can still write to the filesystem holding dir. It
// is a variable so tests can fix it.
var freeBytes = func(dir string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), nil
}

// onFAT reports whether dir is on a FAT filesystem.
func onFAT(dir string) bool {
	var st syscall.Statfs_t
	return syscall.Statfs(dir, &st) == nil && int64(st.Type) == msdosSuperMagic
}

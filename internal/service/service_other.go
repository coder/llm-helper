//go:build !linux && !darwin

package service

import (
	"errors"
	"os/exec"
)

var errUnsupported = errors.New("service management is only implemented on Linux (systemd) and macOS (launchd)")

func (s Service) IsActive() string { return "unknown" }

func (s Service) Label() string { return "service" }

func (s Service) NRestarts() int { return -1 }

func (s Service) Health() Health { return Health{State: "unknown", NRestarts: -1} }

func (s Service) Restart() error { return errUnsupported }

func (s Service) Stop() error { return errUnsupported }

func (s Service) Start() error { return errUnsupported }

func (s Service) StartTail(int) (*exec.Cmd, error) { return nil, errUnsupported }

func (s Service) PrintLogs(int) error { return errUnsupported }

func (s Service) LastLogs(int) string { return "" }

func (s Service) LogHint() string { return "" }

func (s Service) Installed() bool { return false }

func (s Service) Install(string) (string, error) { return "", errUnsupported }

func (s Service) Uninstall() error { return errUnsupported }

func InstanceName(profile string) string { return profile }

// RouterName is the name the model router would run under.
const RouterName = "llm-helper-router"

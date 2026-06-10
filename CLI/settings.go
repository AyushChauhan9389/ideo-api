package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// Settings is a flat key→value store persisted as JSON in the user config dir
// (Windows: %AppData%\ideogram4\config.json, Linux: ~/.config/ideogram4/...).
type Settings map[string]string

var knownSettings = map[string]string{
	"url":              "Base URL of the Ideogram 4 API server (e.g. http://127.0.0.1:8000)",
	"magic_prompt_key": "Default magic-prompt API key sent with generate requests",
	"hive_text_key":    "Default Hive text-moderation key sent with generate requests",
	"hive_visual_key":  "Default Hive visual-moderation key sent with generate requests",
	"default_preset":   "Default sampler preset for generate (e.g. V4_QUALITY_48)",
	"download_dir":     "Default directory for downloaded images (default: current dir)",
	"timeout_seconds":  "HTTP timeout for generate requests (default: 1800)",
}

func settingsPath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("cannot locate user config dir: %w", err)
	}
	return filepath.Join(dir, "ideogram4", "config.json"), nil
}

func loadSettings() Settings {
	s := Settings{}
	path, err := settingsPath()
	if err != nil {
		return s
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return s
	}
	_ = json.Unmarshal(data, &s)
	return s
}

func (s Settings) save() error {
	path, err := settingsPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

func settingsUsage() {
	fmt.Fprint(os.Stderr, `Manage CLI settings.

Usage:
  ideogram4-cli settings list                 Show all settings
  ideogram4-cli settings get <key>            Show one setting
  ideogram4-cli settings set <key> <value>    Set a setting
  ideogram4-cli settings unset <key>          Remove a setting
  ideogram4-cli settings path                 Show the config file location

Known keys:
`)
	keys := make([]string, 0, len(knownSettings))
	for k := range knownSettings {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Fprintf(os.Stderr, "  %-18s %s\n", k, knownSettings[k])
	}
}

func cmdSettings(args []string) error {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		settingsUsage()
		return nil
	}
	s := loadSettings()
	switch args[0] {
	case "list":
		if len(s) == 0 {
			fmt.Println("(no settings; run 'settings set url http://127.0.0.1:8000' to start)")
			return nil
		}
		keys := make([]string, 0, len(s))
		for k := range s {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Printf("%s = %s\n", k, maskIfSecret(k, s[k]))
		}
		return nil
	case "get":
		if len(args) != 2 {
			return fmt.Errorf("usage: settings get <key>")
		}
		v, ok := s[args[1]]
		if !ok {
			return fmt.Errorf("setting %q is not set", args[1])
		}
		fmt.Println(v)
		return nil
	case "set":
		if len(args) != 3 {
			return fmt.Errorf("usage: settings set <key> <value>")
		}
		key, value := args[1], args[2]
		if _, known := knownSettings[key]; !known {
			fmt.Fprintf(os.Stderr, "warning: %q is not a known setting (storing anyway)\n", key)
		}
		s[key] = value
		if err := s.save(); err != nil {
			return err
		}
		fmt.Printf("%s = %s\n", key, maskIfSecret(key, value))
		return nil
	case "unset":
		if len(args) != 2 {
			return fmt.Errorf("usage: settings unset <key>")
		}
		delete(s, args[1])
		if err := s.save(); err != nil {
			return err
		}
		fmt.Printf("unset %s\n", args[1])
		return nil
	case "path":
		path, err := settingsPath()
		if err != nil {
			return err
		}
		fmt.Println(path)
		return nil
	default:
		settingsUsage()
		return fmt.Errorf("unknown settings subcommand %q", args[0])
	}
}

func maskIfSecret(key, value string) string {
	switch key {
	case "magic_prompt_key", "hive_text_key", "hive_visual_key":
		if len(value) > 8 {
			return value[:4] + "..." + value[len(value)-4:]
		}
		return "****"
	}
	return value
}

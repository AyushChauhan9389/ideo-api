package main

import (
	"flag"
	"fmt"
	"time"
)

// newInfoFlagSet builds the shared flag set for the simple read-only commands.
func newInfoFlagSet(name string, args []string) (*flag.FlagSet, *string, *bool, error) {
	fs := flag.NewFlagSet(name, flag.ExitOnError)
	urlFlag := fs.String("url", "", "API base URL (overrides env + settings)")
	asJSON := fs.Bool("json", false, "print raw JSON")
	err := fs.Parse(args)
	return fs, urlFlag, asJSON, err
}

func cmdHealth(args []string) error {
	_, urlFlag, _, err := newInfoFlagSet("health", args)
	if err != nil {
		return err
	}
	client := newClient(*urlFlag, loadSettings(), 30*time.Second)
	var out map[string]any
	if err := client.getJSON("/health", &out); err != nil {
		return err
	}
	return printJSON(out)
}

func cmdPresets(args []string) error {
	_, urlFlag, asJSON, err := newInfoFlagSet("presets", args)
	if err != nil {
		return err
	}
	client := newClient(*urlFlag, loadSettings(), 30*time.Second)
	var out map[string]struct {
		NumSteps         int       `json:"num_steps"`
		GuidanceSchedule []float64 `json:"guidance_schedule"`
		Mu               float64   `json:"mu"`
		Std              float64   `json:"std"`
	}
	if err := client.getJSON("/presets", &out); err != nil {
		return err
	}
	if *asJSON {
		return printJSON(out)
	}
	for name, p := range out {
		fmt.Printf("%-16s steps=%-3d mu=%-5.2f std=%-5.2f schedule(loop-index order)=%v\n",
			name, p.NumSteps, p.Mu, p.Std, summarizeSchedule(p.GuidanceSchedule))
	}
	return nil
}

func cmdMagicModels(args []string) error {
	_, urlFlag, _, err := newInfoFlagSet("magic-models", args)
	if err != nil {
		return err
	}
	client := newClient(*urlFlag, loadSettings(), 30*time.Second)
	var out map[string]any
	if err := client.getJSON("/magic-prompt-models", &out); err != nil {
		return err
	}
	return printJSON(out)
}

// summarizeSchedule compresses runs like [3 3 3 7 7 ...] into "3x3, 7x45".
func summarizeSchedule(s []float64) string {
	if len(s) == 0 {
		return "[]"
	}
	out := ""
	run, count := s[0], 1
	flush := func() {
		if out != "" {
			out += ", "
		}
		out += fmt.Sprintf("%gx%d", run, count)
	}
	for _, v := range s[1:] {
		if v == run {
			count++
			continue
		}
		flush()
		run, count = v, 1
	}
	flush()
	return "[" + out + "]"
}

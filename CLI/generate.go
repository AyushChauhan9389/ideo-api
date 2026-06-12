package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// stringSlice lets -prompt be passed multiple times for batch generation.
type stringSlice []string

func (s *stringSlice) String() string { return strings.Join(*s, ", ") }
func (s *stringSlice) Set(v string) error {
	*s = append(*s, v)
	return nil
}

type generateResponse struct {
	ID               string   `json:"id"`
	CreatedAt        string   `json:"created_at"`
	Seed             int64    `json:"seed"`
	Prompts          []string `json:"prompts"`
	ExpandedCaptions []string `json:"expanded_captions"`
	Images           []struct {
		Filename     string `json:"filename"`
		URL          string `json:"url"`
		PromptIndex  int    `json:"prompt_index"`
		NobgFilename string `json:"nobg_filename"`
		NobgURL      string `json:"nobg_url"`
	} `json:"images"`
	RejectedByModeration []any `json:"rejected_by_moderation"`
	TimingS              struct {
		ModelLoad  float64 `json:"model_load"`
		Generation float64 `json:"generation"`
	} `json:"timing_s"`
}

func cmdGenerate(args []string) error {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "Generate image(s) via POST /generate. Flags map 1:1 to the API body;")
		fmt.Fprintln(os.Stderr, "only flags you pass explicitly are sent, so server defaults apply otherwise.")
		fmt.Fprintln(os.Stderr, "\nUsage: ideogram4-cli generate -prompt \"...\" [flags]")
		fs.PrintDefaults()
	}

	var prompts stringSlice
	fs.Var(&prompts, "prompt", "prompt text (repeat the flag for batch generation; one image per prompt)")
	var promptFiles stringSlice
	fs.Var(&promptFiles, "prompt-file", "read a prompt from a file (useful for structured JSON captions that are painful to shell-quote; repeatable)")

	// Image / sampler
	width := fs.Int("width", 1024, "image width (256-2048, multiple of 16)")
	height := fs.Int("height", 1024, "image height (256-2048, multiple of 16)")
	preset := fs.String("preset", "", "sampler preset (V4_QUALITY_48 | V4_DEFAULT_20 | V4_TURBO_12); empty string = manual sampler config")
	steps := fs.Int("steps", 0, "override the preset's step count (num_steps)")
	guidance := fs.Float64("guidance", 0, "constant CFG weight (guidance_scale); overrides the preset schedule")
	guidanceSchedule := fs.String("guidance-schedule", "", "comma-separated per-step CFG weights in loop-index order, e.g. \"3,3,3,7,7,...\" (length must equal steps)")
	mu := fs.Float64("mu", 0, "logit-normal schedule mean override")
	std := fs.Float64("std", 0, "logit-normal schedule std override")
	seed := fs.Int64("seed", 0, "random seed (omit for a random seed, echoed back in the response)")

	// Magic prompt
	magic := fs.Bool("magic", true, "expand the prompt via a magic-prompt LLM (magic_prompt)")
	magicModel := fs.String("magic-model", "", "magic-prompt configuration (ideogram-4-v1 | claude-opus-v1 | claude-sonnet-v1)")
	magicKey := fs.String("magic-key", "", "magic-prompt API key (falls back to the magic_prompt_key setting, then server env)")
	stripBboxes := fs.Bool("strip-bboxes", true, "strip bbox layout hints from the expanded caption")
	raiseCaptionIssues := fs.Bool("raise-caption-issues", true, "reject prompts the caption verifier flags (false = warn only)")

	// Model selection
	quant := fs.String("quant", "", "weight quantization: nf4 | fp8 (reloads the server pipeline if changed)")
	device := fs.String("device", "", "device override, e.g. cuda, cuda:1, cpu, mps")
	dtype := fs.String("dtype", "", "compute dtype: bfloat16 | float16 | float32")

	// Background removal
	removeBg := fs.Bool("remove-bg", false, "also produce a transparent-background version of each image (remove_background)")
	bgModel := fs.String("bg-model", "", "background-removal model: birefnet-hr (default, quality) | birefnet (fast)")

	// Safety
	hiveTextKey := fs.String("hive-text-key", "", "Hive text moderation key (falls back to the hive_text_key setting)")
	hiveVisualKey := fs.String("hive-visual-key", "", "Hive visual moderation key (falls back to the hive_visual_key setting)")
	moderateInput := fs.Bool("moderate-input", true, "screen prompts with Hive when a key is available")
	moderateOutput := fs.Bool("moderate-output", true, "screen generated images with Hive when a key is available")

	// CLI behavior
	download := fs.Bool("download", false, "download the generated image(s) locally after generation")
	outDir := fs.String("out", "", "directory for downloaded images (default: download_dir setting, else current dir)")
	asJSON := fs.Bool("json", false, "print the full API response as JSON")
	urlFlag := fs.String("url", "", "API base URL (overrides env + settings)")
	timeout := fs.Duration("timeout", 0, "HTTP timeout (default: timeout_seconds setting, else 30m)")

	if err := fs.Parse(args); err != nil {
		return err
	}
	for _, pf := range promptFiles {
		data, err := os.ReadFile(pf)
		if err != nil {
			return fmt.Errorf("reading -prompt-file: %w", err)
		}
		// Strip a UTF-8 BOM: PowerShell's Out-File -Encoding utf8 writes one,
		// and it breaks server-side JSON caption parsing.
		text := strings.TrimPrefix(string(data), "\uFEFF")
		prompts = append(prompts, strings.TrimSpace(text))
	}
	if len(prompts) == 0 && fs.NArg() > 0 {
		prompts = append(prompts, strings.Join(fs.Args(), " "))
	}
	if len(prompts) == 0 {
		fs.Usage()
		return fmt.Errorf("at least one -prompt or -prompt-file is required")
	}

	visited := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { visited[f.Name] = true })

	settings := loadSettings()

	body := map[string]any{}
	if len(prompts) == 1 {
		body["prompt"] = prompts[0]
	} else {
		body["prompt"] = []string(prompts)
	}
	if visited["width"] {
		body["width"] = *width
	}
	if visited["height"] {
		body["height"] = *height
	}
	switch {
	case visited["preset"] && *preset == "":
		body["sampler_preset"] = nil
	case visited["preset"]:
		body["sampler_preset"] = *preset
	case settings["default_preset"] != "":
		body["sampler_preset"] = settings["default_preset"]
	}
	if visited["steps"] {
		body["num_steps"] = *steps
	}
	if visited["guidance"] {
		body["guidance_scale"] = *guidance
	}
	if visited["guidance-schedule"] {
		schedule, err := parseFloats(*guidanceSchedule)
		if err != nil {
			return fmt.Errorf("invalid -guidance-schedule: %w", err)
		}
		body["guidance_schedule"] = schedule
	}
	if visited["mu"] {
		body["mu"] = *mu
	}
	if visited["std"] {
		body["std"] = *std
	}
	if visited["seed"] {
		body["seed"] = *seed
	}
	if visited["magic"] {
		body["magic_prompt"] = *magic
	}
	if visited["magic-model"] {
		body["magic_prompt_model"] = *magicModel
	}
	if key := firstNonEmpty(*magicKey, settings["magic_prompt_key"]); key != "" {
		body["magic_prompt_key"] = key
	}
	if visited["strip-bboxes"] {
		body["strip_bboxes"] = *stripBboxes
	}
	if visited["raise-caption-issues"] {
		body["raise_on_caption_issues"] = *raiseCaptionIssues
	}
	if visited["quant"] {
		body["quantization"] = *quant
	}
	if visited["device"] {
		body["device"] = *device
	}
	if visited["dtype"] {
		body["dtype"] = *dtype
	}
	if visited["remove-bg"] {
		body["remove_background"] = *removeBg
	}
	if visited["bg-model"] {
		body["bg_model"] = *bgModel
	}
	if key := firstNonEmpty(*hiveTextKey, settings["hive_text_key"]); key != "" {
		body["hive_text_key"] = key
	}
	if key := firstNonEmpty(*hiveVisualKey, settings["hive_visual_key"]); key != "" {
		body["hive_visual_key"] = key
	}
	if visited["moderate-input"] {
		body["moderate_input"] = *moderateInput
	}
	if visited["moderate-output"] {
		body["moderate_output"] = *moderateOutput
	}

	t := *timeout
	if t == 0 {
		t = settingsTimeout(settings)
	}
	client := newClient(*urlFlag, settings, t)

	fmt.Fprintf(os.Stderr, "generating %d image(s)... (this can take a while on first run: model download + load)\n", len(prompts))
	var resp generateResponse
	if err := client.postJSON("/generate", body, &resp); err != nil {
		return err
	}

	if *asJSON {
		return printJSON(resp)
	}

	fmt.Printf("id:      %s\n", resp.ID)
	fmt.Printf("seed:    %d\n", resp.Seed)
	fmt.Printf("timing:  model_load=%.1fs generation=%.1fs\n", resp.TimingS.ModelLoad, resp.TimingS.Generation)
	for _, img := range resp.Images {
		fmt.Printf("image:   %s  (%s%s)\n", img.Filename, client.base, img.URL)
		if img.NobgURL != "" {
			fmt.Printf("no-bg:   %s  (%s%s)\n", img.NobgFilename, client.base, img.NobgURL)
		}
	}
	if len(resp.RejectedByModeration) > 0 {
		fmt.Printf("rejected by moderation: %d image(s)\n", len(resp.RejectedByModeration))
	}

	if *download {
		dir := firstNonEmpty(*outDir, settings["download_dir"], ".")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
		for _, img := range resp.Images {
			dest := filepath.Join(dir, img.Filename)
			if err := client.downloadFile(img.URL, dest); err != nil {
				return err
			}
			fmt.Printf("saved:   %s\n", dest)
			if img.NobgURL != "" {
				nobgDest := filepath.Join(dir, img.NobgFilename)
				if err := client.downloadFile(img.NobgURL, nobgDest); err != nil {
					return err
				}
				fmt.Printf("saved:   %s\n", nobgDest)
			}
		}
	}
	return nil
}

func parseFloats(s string) ([]float64, error) {
	parts := strings.Split(s, ",")
	out := make([]float64, 0, len(parts))
	for _, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

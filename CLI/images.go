package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func imagesUsage() {
	fmt.Fprint(os.Stderr, `Manage generated images on the server.

Usage:
  ideogram4-cli images list [-limit N] [-offset N] [-json]      List images (newest first)
  ideogram4-cli images get <filename> [-o path]                 Download an image
  ideogram4-cli images meta <filename>                          Show an image's generation metadata
  ideogram4-cli images delete <filename>                        Delete an image (and its metadata)

Common flags: -url <base-url>
`)
}

type imageList struct {
	Total  int `json:"total"`
	Limit  int `json:"limit"`
	Offset int `json:"offset"`
	Images []struct {
		Filename   string `json:"filename"`
		URL        string `json:"url"`
		SizeBytes  int64  `json:"size_bytes"`
		ModifiedAt string `json:"modified_at"`
		Metadata   *struct {
			Seed    int64    `json:"seed"`
			Prompts []string `json:"prompts"`
		} `json:"metadata"`
	} `json:"images"`
}

func cmdImages(args []string) error {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		imagesUsage()
		return nil
	}
	sub, rest := args[0], args[1:]
	settings := loadSettings()

	// Go's flag package stops at the first positional arg, so support the
	// natural "images get <filename> -o path" order by popping the filename
	// before parsing flags.
	popFilename := func() (string, []string) {
		if len(rest) > 0 && !strings.HasPrefix(rest[0], "-") {
			return rest[0], rest[1:]
		}
		return "", rest
	}

	switch sub {
	case "list":
		fs := flag.NewFlagSet("images list", flag.ExitOnError)
		limit := fs.Int("limit", 50, "max images to list")
		offset := fs.Int("offset", 0, "pagination offset")
		asJSON := fs.Bool("json", false, "print raw JSON")
		urlFlag := fs.String("url", "", "API base URL")
		if err := fs.Parse(rest); err != nil {
			return err
		}
		client := newClient(*urlFlag, settings, 60*time.Second)
		var list imageList
		if err := client.getJSON(fmt.Sprintf("/images?limit=%d&offset=%d", *limit, *offset), &list); err != nil {
			return err
		}
		if *asJSON {
			return printJSON(list)
		}
		if list.Total == 0 {
			fmt.Println("no generated images yet")
			return nil
		}
		fmt.Printf("%d image(s) total, showing %d:\n\n", list.Total, len(list.Images))
		for _, img := range list.Images {
			prompt := ""
			if img.Metadata != nil && len(img.Metadata.Prompts) > 0 {
				prompt = img.Metadata.Prompts[0]
				if len(prompt) > 60 {
					prompt = prompt[:57] + "..."
				}
			}
			fmt.Printf("%-44s %8.1f KB  %s  %s\n", img.Filename, float64(img.SizeBytes)/1024, img.ModifiedAt, prompt)
		}
		return nil

	case "get":
		filename, getArgs := popFilename()
		fs := flag.NewFlagSet("images get", flag.ExitOnError)
		output := fs.String("o", "", "output path (default: download_dir setting + original filename)")
		urlFlag := fs.String("url", "", "API base URL")
		if err := fs.Parse(getArgs); err != nil {
			return err
		}
		if filename == "" && fs.NArg() == 1 {
			filename = fs.Arg(0)
		}
		if filename == "" {
			return fmt.Errorf("usage: images get <filename> [-o path]")
		}
		dest := *output
		if dest == "" {
			dest = filepath.Join(firstNonEmpty(settings["download_dir"], "."), filename)
		}
		if dir := filepath.Dir(dest); dir != "." {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return err
			}
		}
		client := newClient(*urlFlag, settings, 5*time.Minute)
		if err := client.downloadFile("/images/"+filename, dest); err != nil {
			return err
		}
		fmt.Printf("saved: %s\n", dest)
		return nil

	case "meta":
		filename, metaArgs := popFilename()
		fs := flag.NewFlagSet("images meta", flag.ExitOnError)
		urlFlag := fs.String("url", "", "API base URL")
		if err := fs.Parse(metaArgs); err != nil {
			return err
		}
		if filename == "" && fs.NArg() == 1 {
			filename = fs.Arg(0)
		}
		if filename == "" {
			return fmt.Errorf("usage: images meta <filename>")
		}
		client := newClient(*urlFlag, settings, 60*time.Second)
		var meta map[string]any
		if err := client.getJSON("/images/"+filename+"/metadata", &meta); err != nil {
			return err
		}
		return printJSON(meta)

	case "delete":
		filename, delArgs := popFilename()
		fs := flag.NewFlagSet("images delete", flag.ExitOnError)
		urlFlag := fs.String("url", "", "API base URL")
		if err := fs.Parse(delArgs); err != nil {
			return err
		}
		if filename == "" && fs.NArg() == 1 {
			filename = fs.Arg(0)
		}
		if filename == "" {
			return fmt.Errorf("usage: images delete <filename>")
		}
		client := newClient(*urlFlag, settings, 60*time.Second)
		if _, err := client.deletePath("/images/" + filename); err != nil {
			return err
		}
		fmt.Printf("deleted: %s\n", filename)
		return nil

	default:
		imagesUsage()
		return fmt.Errorf("unknown images subcommand %q", sub)
	}
}

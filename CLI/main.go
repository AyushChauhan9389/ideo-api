// ideogram4-cli — command-line client for the Ideogram 4 API server (api_server.py).
//
// Configure once:
//
//	ideogram4-cli settings set url http://127.0.0.1:8000
//
// Then:
//
//	ideogram4-cli generate -prompt "a ginger cat wearing a tiny wizard hat"
//	ideogram4-cli images list
//	ideogram4-cli images get 20260611_120000_abc123_0.png -o cat.png
package main

import (
	"fmt"
	"os"
)

const version = "0.1.0"

func usage() {
	fmt.Fprint(os.Stderr, `ideogram4-cli `+version+` — client for the Ideogram 4 API server

Usage:
  ideogram4-cli <command> [flags]

Commands:
  generate       Generate image(s) from a prompt (every API parameter exposed as a flag)
  images         Manage generated images: list, get, meta, delete
  presets        Show the server's named sampler presets
  magic-models   Show available magic-prompt configurations
  health         Show server status and the currently loaded pipeline
  settings       Manage CLI settings (API url, default keys, ...)
  version        Print the CLI version
  help           Show this help

Run 'ideogram4-cli <command> -h' for command-specific flags.

The API base URL is resolved in this order:
  1. -url flag    2. IDEOGRAM4_URL env var    3. 'settings set url ...'    4. http://127.0.0.1:8000
`)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd, args := os.Args[1], os.Args[2:]

	var err error
	switch cmd {
	case "generate":
		err = cmdGenerate(args)
	case "images":
		err = cmdImages(args)
	case "presets":
		err = cmdPresets(args)
	case "magic-models":
		err = cmdMagicModels(args)
	case "health":
		err = cmdHealth(args)
	case "settings":
		err = cmdSettings(args)
	case "version":
		fmt.Println(version)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", cmd)
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

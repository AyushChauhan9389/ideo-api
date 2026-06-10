package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const defaultBaseURL = "http://127.0.0.1:8000"

// Client is a thin HTTP wrapper over the Ideogram 4 API server.
type Client struct {
	base string
	http *http.Client
}

// newClient resolves the base URL: -url flag > IDEOGRAM4_URL env > settings > default.
func newClient(urlFlag string, settings Settings, timeout time.Duration) *Client {
	base := urlFlag
	if base == "" {
		base = os.Getenv("IDEOGRAM4_URL")
	}
	if base == "" {
		base = settings["url"]
	}
	if base == "" {
		base = defaultBaseURL
	}
	return &Client{
		base: strings.TrimRight(base, "/"),
		http: &http.Client{Timeout: timeout},
	}
}

// settingsTimeout returns the configured generate timeout (default 30 min).
func settingsTimeout(settings Settings) time.Duration {
	if v := settings["timeout_seconds"]; v != "" {
		if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
			return time.Duration(secs) * time.Second
		}
	}
	return 30 * time.Minute
}

type apiError struct {
	Status int
	Detail any
}

func (e *apiError) Error() string {
	detail, err := json.MarshalIndent(e.Detail, "", "  ")
	if err != nil {
		return fmt.Sprintf("API error %d", e.Status)
	}
	return fmt.Sprintf("API error %d: %s", e.Status, detail)
}

func (c *Client) do(method, path string, body any) ([]byte, error) {
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(data)
	}
	req, err := http.NewRequest(method, c.base+path, reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request to %s failed: %w (is the API server running? set the url with 'settings set url ...')", c.base, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var parsed struct {
			Detail any `json:"detail"`
		}
		if json.Unmarshal(data, &parsed) == nil && parsed.Detail != nil {
			return nil, &apiError{Status: resp.StatusCode, Detail: parsed.Detail}
		}
		return nil, &apiError{Status: resp.StatusCode, Detail: strings.TrimSpace(string(data))}
	}
	return data, nil
}

func (c *Client) getJSON(path string, out any) error {
	data, err := c.do(http.MethodGet, path, nil)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, out)
}

func (c *Client) postJSON(path string, body any, out any) error {
	data, err := c.do(http.MethodPost, path, body)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, out)
}

func (c *Client) deletePath(path string) ([]byte, error) {
	return c.do(http.MethodDelete, path, nil)
}

// downloadFile streams GET <path> into dest.
func (c *Client) downloadFile(path, dest string) error {
	resp, err := c.http.Get(c.base + path)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("download %s: HTTP %d: %s", path, resp.StatusCode, strings.TrimSpace(string(data)))
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

// printJSON pretty-prints any JSON-marshalable value to stdout.
func printJSON(v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(data))
	return nil
}

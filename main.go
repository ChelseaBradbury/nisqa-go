// Command nisqa scores speech quality with the NISQA model.
//
// It is a Go port of the Python proof of concept in ../nisqa-poc: same model,
// same features, same numbers — but the front-end is reimplemented in Go and
// the network runs through ONNX Runtime instead of PyTorch.
//
//	nisqa clip.wav
//	nisqa samples/
//	nisqa a.wav b.wav --csv out.csv
package main

import (
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"nisqa-go/internal/audio"
	"nisqa-go/internal/nisqa"
)

//go:embed models/*.onnx models/*.json
var models embed.FS

// Column order for display, and what each dimension means. The network emits
// mos, noi, dis, col, loud; this is the order the Python POC prints.
var display = []struct{ key, header, meaning string }{
	{"mos", "MOS", "overall quality"},
	{"noi", "NOI", "noisiness"},
	{"col", "COL", "coloration"},
	{"dis", "DIS", "discontinuity"},
	{"loud", "LOUD", "loudness"},
}

var audioExts = map[string]bool{".wav": true}

type result struct {
	path   string
	scores map[string]float32
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	model := flag.String("model", "dim", "which exported NISQA model to use")
	csvPath := flag.String("csv", "", "also write the predictions to this csv")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "usage: %s [flags] <audio file or folder>...\n\nflags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	inputs, err := parseArgs()
	if err != nil {
		return err
	}
	if len(inputs) == 0 {
		flag.Usage()
		return fmt.Errorf("no input given")
	}

	files, err := collectFiles(inputs)
	if err != nil {
		return err
	}

	onnxData, err := models.ReadFile("models/nisqa_" + *model + ".onnx")
	if err != nil {
		return fmt.Errorf("no exported model %q (run tools/export_onnx.py --model %s)", *model, *model)
	}
	metaData, err := models.ReadFile("models/nisqa_" + *model + ".json")
	if err != nil {
		return err
	}
	meta, err := nisqa.ParseMeta(metaData)
	if err != nil {
		return err
	}

	predictor, err := nisqa.New(onnxData, meta)
	if err != nil {
		return err
	}
	defer predictor.Close()

	results := make([]result, 0, len(files))
	for _, path := range files {
		clip, err := audio.DecodeWAV(path)
		if err != nil {
			return err
		}
		scores, err := predictor.Score(clip)
		if err != nil {
			return fmt.Errorf("%s: %w", filepath.Base(path), err)
		}
		byName := make(map[string]float32, len(scores))
		for i, name := range meta.Outputs {
			byName[name] = scores[i]
		}
		results = append(results, result{path: path, scores: byName})
	}

	printTable(results, meta)
	if *csvPath != "" {
		if err := writeCSV(*csvPath, results, meta); err != nil {
			return err
		}
		fmt.Printf("wrote %s\n", *csvPath)
	}
	return nil
}

// parseArgs is flag.Parse, but tolerant of flags appearing after filenames, so
// "nisqa a.wav b.wav --csv out.csv" works as well as the strict flags-first form.
func parseArgs() ([]string, error) {
	if err := flag.CommandLine.Parse(os.Args[1:]); err != nil {
		return nil, err
	}
	var inputs []string
	for rest := flag.Args(); len(rest) > 0; rest = flag.Args() {
		inputs = append(inputs, rest[0])
		if err := flag.CommandLine.Parse(rest[1:]); err != nil {
			return nil, err
		}
	}
	return inputs, nil
}

func collectFiles(args []string) ([]string, error) {
	var files []string
	for _, arg := range args {
		info, err := os.Stat(arg)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() {
			files = append(files, arg)
			continue
		}
		entries, err := os.ReadDir(arg)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if !e.IsDir() && audioExts[strings.ToLower(filepath.Ext(e.Name()))] {
				files = append(files, filepath.Join(arg, e.Name()))
			}
		}
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no .wav files found")
	}
	sort.Strings(files)
	return files, nil
}

// columns returns the display columns the loaded model actually produces.
func columns(meta nisqa.Meta) []struct{ key, header, meaning string } {
	present := make(map[string]bool, len(meta.Outputs))
	for _, name := range meta.Outputs {
		present[name] = true
	}
	var cols []struct{ key, header, meaning string }
	for _, c := range display {
		if present[c.key] {
			cols = append(cols, c)
		}
	}
	return cols
}

func printTable(results []result, meta nisqa.Meta) {
	cols := columns(meta)
	width := len("file")
	for _, r := range results {
		if n := len(filepath.Base(r.path)); n > width {
			width = n
		}
	}

	var header strings.Builder
	fmt.Fprintf(&header, "%-*s", width, "file")
	for _, c := range cols {
		fmt.Fprintf(&header, "  %5s", c.header)
	}
	rule := strings.Repeat("-", header.Len())

	fmt.Printf("\nNISQA (%s) — Go port\n", meta.Model)
	fmt.Println(rule)
	fmt.Println(header.String())
	fmt.Println(rule)
	for _, r := range results {
		fmt.Printf("%-*s", width, filepath.Base(r.path))
		for _, c := range cols {
			fmt.Printf("  %5.2f", r.scores[c.key])
		}
		fmt.Println()
	}
	fmt.Println(rule)
	fmt.Println("scale: 1 (bad) - 5 (excellent)")
	if len(cols) > 1 {
		parts := make([]string, len(cols))
		for i, c := range cols {
			parts[i] = c.header + "=" + c.meaning
		}
		fmt.Println(strings.Join(parts, ", "))
	}
	fmt.Println()
}

func writeCSV(path string, results []result, meta nisqa.Meta) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := append([]string{"deg"}, meta.Outputs...)
	if err := w.Write(header); err != nil {
		return err
	}
	for _, r := range results {
		row := make([]string, 0, len(header))
		row = append(row, r.path)
		for _, name := range meta.Outputs {
			row = append(row, strconv.FormatFloat(float64(r.scores[name]), 'f', 6, 32))
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return w.Error()
}

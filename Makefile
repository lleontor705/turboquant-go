.PHONY: build test bench lint vet fmt tidy clean help

GOCMD=go
GOTEST=$(GOCMD) test
GOVET=$(GOCMD) vet
GOFMT=gofmt

help:
	@echo "turboquant-go"
	@echo ""
	@echo "Targets:"
	@echo "  test       Run all tests"
	@echo "  test-race  Run tests with race detector"
	@echo "  bench      Run benchmarks"
	@echo "  vet        Run go vet"
	@echo "  lint       Run golangci-lint"
	@echo "  fmt        Format code"
	@echo "  tidy       Run go mod tidy"
	@echo "  clean      Remove build artifacts"
	@echo "  fixtures   Regenerate test fixtures"

test:
	$(GOTEST) ./...

test-race:
	$(GOTEST) -race ./...

bench:
	$(GOTEST) -bench=. -benchmem ./...

vet:
	$(GOVET) ./...

lint:
	golangci-lint run ./...

fmt:
	$(GOFMT) -s -w .

tidy:
	$(GOCMD) mod tidy

clean:
	@rm -rf dist/

fixtures:
	$(GOCMD) run ./cmd/genfixtures

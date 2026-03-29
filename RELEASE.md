# Release Checklist

Use this checklist to publish a new version.

## 1) Update versioned content

- Verify `README.md` examples are correct.
- Ensure `go.mod` has the correct module path:
  `github.com/lleontor705/turboquant-go`.

## 2) Run validation locally

```bash
go vet ./...
go test ./...
```

## 3) Tag and push

```bash
git tag v0.1.0
git push origin v0.1.0
```

## 4) GitHub Release

- The `release.yml` workflow runs on tag push (`v*`).
- It executes `go vet` + `go test` and creates a GitHub Release with notes.

## 5) pkg.go.dev verification

- Visit: https://pkg.go.dev/github.com/lleontor705/turboquant-go
- If the new version does not appear yet, click **“Request”**.

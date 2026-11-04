# Corpus Audit Report

## Summary

| Metric | Count | Note |
|---|---:|---|
| Source records | 998 | Initial input rows |
| Empty texts removed | 0 | Length = 0 after whitespace |
| Duplicate rows removed | 0 | Identical texts and labels |
| Conflicting groups removed | 0 | Identical texts with conflicting labels |
| Conflicting rows excluded | 0 | Preserved for manual inspection |
| Label aliases normalized | 0 | `literaly` → `literary` |
| **Clean verified records** | **998** | Ready for splitting and modeling |

## Clean Corpus Class Distribution

### Writing Styles

| Style | Count |
|---|---:|
| `academic` | 200 |
| `business` | 199 |
| `formal` | 199 |
| `informal` | 200 |
| `literary` | 200 |

### Communicative Tones

| Tone | Count |
|---|---:|
| `aggressive` | 200 |
| `friendly` | 200 |
| `neutral` | 200 |
| `sarcastic` | 200 |
| `urgent` | 198 |

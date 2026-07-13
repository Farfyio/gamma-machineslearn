# Configuration

Project configuration is loaded through `gamma_machineslearn.config.load_config`.

## Expected YAML shape

```yaml
random_seed: 42
test_size: 0.2
output_dir: outputs
metadata: {}
```

Unknown or dataset-specific fields can be tracked in `metadata` until promoted to explicit config keys.

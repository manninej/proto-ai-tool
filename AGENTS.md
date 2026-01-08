# SAGA Code Agent Notes

## Project identity
- Project: SAGA Code
- CLI: `sage`
- Package: `saga_code`

## Prompt conventions
- Prompts live under `./prompts/<layer>/`.
- Layers are stacked by name using `sage prompts default,fiat,cars`.
- For each bundle and role, the last layer providing `<role>.j2` supplies the body.
- `<role>.prepend.j2` and `<role>.append.j2` are applied for every layer in stack order, even when a later layer overrides the body.
- Shared snippets belong in `./prompts/<layer>/shared/` and may be included with `{% include "shared/example.j2" %}`.
- Optional variables live in `./prompts/<layer>/variables.yaml` and are merged in stack order (later layers override earlier ones).

## Adding bundles or layers
- Create a new bundle directory such as `./prompts/default/new_bundle/`.
- Provide `system.j2` and `user.j2` at minimum.
- Add optional prepend/append fragments as needed.
- To customize prompts, create a new layer under `./prompts/<layer>/` and apply it with `sage prompts`.

## Testing rules
- Do not use the network in tests.
- Keep prompt resolution deterministic (sorted listings, explicit stack order).
- Add or update tests when changing prompt stacking behavior.

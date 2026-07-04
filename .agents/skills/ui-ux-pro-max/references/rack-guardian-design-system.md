# Rack Guardian Design System

## Experience

Rack Guardian is a decision interface for a non-technical GPU-cluster operator. The
experience is a composed technical co-pilot: alert, explanation, recommendation,
consent, and outcome. It is not a monitoring dashboard.

## Information hierarchy

1. Assistant verdict: rack, risk, and urgency in one sentence.
2. Recommended action: explicit source, workload, and destination.
3. Expected outcome: before/after equilibrium and avoided slowdown.
4. Operator consent: one dominant acceptance action.
5. Supporting evidence: explanation, alternatives, and technical detail on demand.

## Tokens

- Canvas: near-black neutral `#070B10`.
- Primary surface: `#0D141D`.
- Raised surface: `#111B26`.
- Primary text: cool white `#F4F8FB`.
- Secondary text: `#A7B4C2`.
- Muted text: `#718096`.
- Assistant accent: cyan `#43C7E8`.
- Critical: red `#F06464`.
- Warning: amber `#E5A94D`.
- Safe: green `#45C58A`.
- Borders: white at 8-12% opacity.

Prefer semantic CSS variables over repeating raw values.

## Type and spacing

- Use the existing system sans stack to avoid network and font loading risk.
- Assistant verdict: 28-36px desktop, 22-28px mobile, 1.25-1.4 line height.
- Section heading: 16-20px, semibold.
- Body: 16px minimum, 1.55-1.7 line height.
- Metadata: 12-14px, never below 12px.
- Use an 8px spacing rhythm with 24-40px separation between major groups.
- Keep the decision column between 720px and 880px.

## Surfaces and depth

- Use one primary assistant stage and at most two compact supporting surfaces.
- Use 6-8px radii for framed controls and cards.
- Borders carry structure; shadows are broad, dark, and very subtle.
- Avoid nested cards and avoid turning every data point into a tile.

## Motion

- Assistant presence: low-amplitude 2.8-4s breathing loop.
- Typing cursor: optional, never delay access to the complete decision excessively.
- State transition: 180-280ms fade/color shift with stable dimensions.
- Modal/panel reveal: 180-240ms opacity plus 4-8px movement.
- Reduced motion: stop decorative loops and reveal content immediately.

## Accessibility and trust

- Pair every semantic color with text and/or an icon.
- Announce assistant message updates with a polite live region.
- Do not auto-speak on page load. Speech begins only after operator intent.
- Keep focus visible and return focus after dismissing a dialog where practical.
- Avoid disabled controls without an explanation or tooltip.
- State clearly that execution is simulated and the operator remains in control.

## Source

Adapted for Rack Guardian from `nextlevelbuilder/ui-ux-pro-max-skill`. The original
MIT license is included at the skill root. Searchable reference data and scripts are
retained for future UI/UX tasks.

---
name: ui-ux-pro-max
description: Use this skill for frontend UI, UX, visual polish, design systems, interaction design, accessibility, layout, responsive design, and demo-ready product interfaces.
---

# UI/UX Pro Max for Rack Guardian

Use this skill to design, audit, or refine Rack Guardian frontend experiences. Read
`references/rack-guardian-design-system.md` before editing the UI. The original
searchable design database is retained under `data/`, with its scripts under
`scripts/`.

## Required workflow

1. Inspect the existing stack, UI structure, states, and current browser rendering.
2. Identify the operator's single most important decision.
3. Audit the first viewport using the five-second test below.
4. Establish or update the small semantic design system in the reference file.
5. Implement the shortest clear path from alert to informed action.
6. Test every interactive state, keyboard focus, reduced motion, and responsive layout.
7. Run the repository's build and lint commands, then fix errors.

For broad design exploration, run:

```powershell
python .agents/skills/ui-ux-pro-max/scripts/search.py "AI operations assistant calm dark trustworthy" --design-system -p "Rack Guardian"
```

For focused guidance:

```powershell
python .agents/skills/ui-ux-pro-max/scripts/search.py "accessibility motion focus dark contrast" --domain ux
python .agents/skills/ui-ux-pro-max/scripts/search.py "rendering state effects" --stack react
```

Treat search results as supporting evidence. Rack Guardian's product rules below
override generic dashboard, SaaS, landing-page, or visual-style suggestions.

## Rack Guardian product rules

- Build a calm AI assistant and mission-control co-pilot, never a dense dashboard.
- Make the situation understandable in under five seconds.
- Present one problem, one plain-language explanation, one recommendation, and one
  primary action.
- Show the consequence before acceptance. Recommendations are advisory, not forced.
- Hide telemetry, confidence, causes, and charts behind progressive disclosure.
- Keep the assistant conversational, voice-friendly, concise, and action-oriented.
- Use a premium dark interface with large typography, strong spacing, restrained
  borders, and one clear visual anchor.
- Reserve red for active critical risk, amber for warning, green for safe outcomes,
  and cyan/blue for assistant presence and action.
- Avoid cheap neon, cyberpunk effects, gaming UI, decorative HUD markings, clutter,
  tiny labels, generic SaaS cards, excessive glass effects, and animated noise.

## Five-second test

The first viewport must answer, in this order:

1. What is happening?
2. How soon does it matter?
3. What does the assistant recommend?
4. What happens if the operator accepts?
5. What is the one primary control?

If any answer requires scrolling, decoding a chart, or comparing multiple metrics,
redesign the hierarchy.

## Interaction rules

- Keep one dominant CTA. Alternatives and technical details remain secondary.
- Give every control a visible hover, focus, pressed, disabled, and success state.
- Use native buttons and semantic regions. Maintain logical reading and tab order.
- Minimum target size is 44 by 44 CSS pixels.
- Animate opacity and transforms only; typical feedback lasts 150-300ms.
- Continuous animation is allowed only for live assistant presence and must be subtle.
- Respect `prefers-reduced-motion`; never make motion necessary for comprehension.
- Voice features must remain optional and fail gracefully without hiding text.
- Preserve stable layout dimensions while messages type or statuses change.

## Visual audit checklist

- Hierarchy: the primary message dominates, while metadata recedes.
- Density: no more than two supporting surfaces in the primary decision area.
- Typography: body text is at least 16px; compact labels remain legible.
- Measure: conversational copy stays within roughly 45-70 characters per line.
- Contrast: body text meets WCAG AA; state colors are never the only signal.
- Spacing: use a consistent 4/8px rhythm and generous section separation.
- Color: neutral surfaces dominate; semantic color appears only where meaningful.
- Motion: no layout shift, long entrance choreography, or competing loops.
- Responsive: validate 375px mobile and common desktop widths without overflow.
- Content: use calm, direct sentences and explain operational consequences.

## Delivery standard

The finished screen should feel credible in a live hackathon demo without requiring
the presenter to explain the interface. Preserve function while improving clarity.
Validate the initial alert, explanation, alternatives, technical details, voice
fallback, and accepted/safe outcome before completion.


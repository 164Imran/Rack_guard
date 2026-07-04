# Rack Guardian Repository Instructions

## Product Direction

Rack Guardian is an AI thermal co-pilot for GPU cluster operators. It is a calm
decision assistant, not a monitoring dashboard.

Every primary screen must be understandable by a non-technical operator in under
five seconds. Prioritize demo clarity over feature quantity.

## Required UI Hierarchy

Before changing the UI, plan the screen hierarchy around this sequence:

1. One alert: what is happening and how urgent it is.
2. One explanation: why it is happening, in plain language.
3. One recommendation: what the assistant proposes and its expected consequence.
4. One primary action: the operator explicitly accepts or rejects the recommendation.

Keep the experience conversational and assistant-driven. Make it clear that the AI
is recommending an action and that the operator remains in control.

## Progressive Disclosure

- Hide technical telemetry, confidence scores, detailed causes, and charts by default.
- Reveal technical evidence only when the operator requests it.
- Keep alternatives secondary to the recommended action.
- Never require a graph or dense metric comparison to understand the decision.

## Visual and Interaction Design

- Use a premium, sober, accessible dark interface.
- Use large readable typography, clear hierarchy, generous spacing, and restrained motion.
- Reserve semantic colors for meaning: red for critical, amber for warning, green for
  safe outcomes, and cyan or blue for assistant presence and actions.
- Provide visible keyboard focus, sufficient contrast, semantic controls, and
  reduced-motion support.
- Keep voice features optional and preserve complete text alternatives.
- Avoid cheap neon, clutter, dashboard overload, tiny text, excessive cards, gaming
  UI, decorative telemetry, and cyberpunk cliches.

## Implementation

- Preserve the existing React, TypeScript, and Tailwind stack unless the task requires otherwise.
- Keep state and components focused; do not introduce unnecessary dependencies.
- Preserve all existing operator workflows when refining the visual design.
- Verify responsive behavior and all relevant interaction states after UI changes.
- Run the available build and lint commands after changes and fix any errors before completion.


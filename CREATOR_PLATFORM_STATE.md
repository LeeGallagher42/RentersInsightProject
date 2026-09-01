# Creator Platform Build State

Canonical public build location: `creator-platform/index.html` on branch `creator-platform`.

Current product name: **INKSIDE**.

Implemented before run 10:
- Responsive public landing/discovery experience with mobile bottom navigation.
- Eight seeded original demo art/comic works generated entirely with CSS shapes/gradients.
- Search, format filtering, featured/new/popular/price sorting, and persistent Saved-only discovery.
- Work-detail modal with content notes, purchase intent, sharing, likes, saves and three-page comic reader demo.
- Creator cards plus focused creator profiles/catalogues with persistent follow state.
- Creator Studio overview metrics, catalogue management and local unpublish action.
- Art/comic publish flow with browser draft autosave, rights confirmation, publishing-rules confirmation, tags, description and optional content note.
- Transparent 90/10 creator revenue positioning with live split preview during publish and purchase intent.
- Basic report categories, optional report detail, local moderation queue and report timestamps/status.
- Analytics-ready product event queue persisted to localStorage under `inkside_analytics`.
- Persistent demo likes/follows/saves/published works/reports via localStorage.
- Keyboard card activation, visible focus states, Escape-to-close modal behavior and responsive breakpoints.
- Demo content is explicitly rights-safe/original and no third-party art assets are embedded.

Run 10 product upgrade preserved as a compressed candidate artifact at `creator-platform/run-10-candidate.html.gz.b64` (gzip-compressed HTML encoded as base64).

Run 10 adds and verifies:
- `prefers-reduced-motion` handling for accessibility.
- Modal focus trapping and focus restoration rather than Escape-only behavior.
- Editable creator profile fields persisted locally (display name, handle, bio, avatar initials).
- Comic-specific page ordering controls with add/remove/move-up/move-down actions before publishing.
- Draft vs published separation in Creator Studio metrics.
- Moderation reviewer actions so queued reports can be marked reviewed, with timestamps/status retained.
- Escaped rendering for user-entered creator/work/report strings to reduce stored-XSS risk in the static MVP.
- 90/10 split maintained in discovery, work detail, publishing and checkout-intent surfaces.
- Seeded demo artwork remains CSS-generated/original with no third-party art assets.

Run 11 candidate is preserved at `creator-platform/run-11-candidate.html.gz.b64`.

Run 11 materially upgrades the product with:
- A redesigned creator-first public landing page and responsive 4/2/1-column discovery layout.
- Eight seeded original CSS-generated artworks/comics with clearer title, creator, pricing, format and mature labels.
- Search plus format, mature-content and sort controls; mature work is hidden by default.
- Rich work detail with comic pagination, creator attribution, content notes, likes, saves, sharing, reporting and checkout-intent events.
- Creator directory and profile modals with persistent follow state and creator catalogues.
- Editable Creator Studio profile plus published/draft/moderation KPIs and catalogue unpublish controls.
- Publish flow with local draft autosave, rights and community-rules confirmations, mature label, tags, content notes and comic page count controls.
- 90/10 revenue split surfaced on the home page, publish preview and purchase-intent view with concrete euro calculations.
- Moderation queue states expanded to reviewed, actioned and dismissed.
- User-entered strings escaped before rendering and analytics events versioned with timestamps.
- Mobile bottom navigation, reduced-motion support, keyboard-openable work cards, modal focus trapping and focus restoration.

Run 11 verification:
- JavaScript parse check: PASS.
- Local HTTP smoke test: HTTP 200.
- Feature smoke checks for Creator Studio, mature-content filtering, publish analytics, comic analytics and 90/10 positioning: PASS.

Run 11 preservation commit: `68f9e26abb475467b534f4d6b1effcd979c8383d`.

Deployment rule: before 2026-09-01 21:00 Europe/Dublin keep improving and verifying. On/after that deadline deploy the best working version and verify the public URL.

Next highest-value work:
1. Promote the best verified candidate into canonical `creator-platform/index.html` after an interaction-level pass.
2. Add explicit terms/community-guidelines/help surfaces and basic empty/error/success states where still thin.
3. Add real upload storage/payment/auth only after the public static MVP is deployed and usable.
4. Verify Vercel project/deployment path and public hosting at the deadline.
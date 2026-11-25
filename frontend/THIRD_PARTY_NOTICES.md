# Third-party notices

## Talizen templates

The RallyMirror interface adapts these MIT-licensed templates:

- [Lunar Rover 3D Scroll](https://github.com/talizen/lunar-rover-3d-scroll), revision `411003681be8ed29f4e57764c570a768844aba85`: staged camera distance/target interpolation, quintic segment easing and frame-rate-independent scroll smoothing from `src/components/rover/engine/scene.ts` and `util.ts`, adapted in `CourtScene.tsx` and `courtScroll.ts`. The rover, moon, textures and global full-page scroll layers are replaced with a section-scoped badminton court.
- [Blue Design Studio](https://github.com/talizen/blue-design-studio), revision `5aa997d4ca198220c30709d047ba135a53b2df62`: text and section entrance sequencing from `src/components/motion.tsx`, `Hero.tsx` and `FeaturedWorks.tsx`, adapted in `EditorialMotion.tsx`. Native browser animations and observers replace GSAP/Lenis; all content stays available with reduced motion or unsupported animation APIs.
- [Aren Vale Composer](https://github.com/talizen/aren-vale-composer), revision `a15386c5c97e3d94e5c04cd86afbe39d51229bed`: spatial image layout, normalised pointer control, gallery/index switching and project links from `src/lib/site/stage.ts`, `home.ts` and `src/page/Projects.tsx`, adapted in `RallyGallery.tsx` as browser-composited CSS geometry. Music, composer imagery and audio are not included.

The project examples use RallyMirror source frames and actual predictions from the team's six selected models. No template testimonials, client claims, property listings or model results are imported. The pasted real-estate reference informs the four-column hero grid and staged typography only.

MIT License

Copyright (c) Talizen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## ThreeUI Community

RallyMirror's `src/components/CourtScene.tsx` adapts the renderer ownership,
resize, observer, visibility and cleanup structure from ThreeUI's
`src/shaders/gallery/Gallery.tsx`, and the pointer-interpolation approach from
`src/shaders/animated-top-dock/glassParticleField.ts`.

Source: https://github.com/MengTo/threeui

Reviewed and adapted revision: `68802d5428071ada5c20db8094b1649e6bb770ed`.

The original gallery content is replaced with new badminton court, net and
shuttle geometry. The adaptation adds lazy loading, capped rendering resolution,
a 30fps draw budget, a static fallback, dynamic reduced-motion support and
explicit WebGL context-loss handling. No Pro/Beta components, remote catalog
media or ThreeUI product branding are redistributed. The open font assets
listed below retain their own licenses.

MIT License

Copyright (c) 2026 Meng To

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Three.js

The application imports Three.js as a separately installed npm dependency. Its
MIT notice is included in the installed package. Three.js is Copyright
2010–2025 three.js authors. See https://github.com/mrdoob/three.js/blob/r180/LICENSE.

## Open fonts

Fragment Mono, Lexend and Instrument Serif font files copied from the ThreeUI
Community assets are redistributed under the SIL Open Font License 1.1. Their
copyright notices and complete license are included in `public/fonts/LICENSE.md`.
No restricted SF Pro font files are included.

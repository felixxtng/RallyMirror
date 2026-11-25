import {
  Component,
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import {
  Activity,
  Aperture,
  ArrowUpRight,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Download,
  ExternalLink,
  Eye,
  FileJson,
  Film,
  Focus,
  GitCompareArrows,
  Grid2X2,
  Layers3,
  LoaderCircle,
  Menu,
  Pause,
  Play,
  Plus,
  ScanLine,
  ShieldCheck,
  Sparkles,
  StepBack,
  StepForward,
  Target,
  Trash2,
  UploadCloud,
  VolumeX,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react';
import { RallyMirrorApi, errorMessage, isAbortError } from './lib/api';
import { loadJobs, saveJob, removeJob } from './lib/jobs';
import {
  ANALYSIS_MODES,
  isAnalysisMode,
  isActiveJob,
  type AnalysisMode,
  type ResultLinks,
  type RunReport,
  type StoredJob,
} from './lib/types';
import { analysisInfo, exampleVideo, examplePoster, exampleReport } from './lib/analysis';
import { finiteNumber, formatBytes, formatTime, videoMetadata } from './utils';
import { EditorialReveal, EditorialText } from './components/EditorialMotion';

const CourtScene = lazy(() => import('./components/CourtScene'));
const RallyGallery = lazy(() => import('./components/RallyGallery'));
const api = new RallyMirrorApi(import.meta.env.VITE_API_BASE_URL || '');
const POSTER = examplePoster('combined');
const stageNames: Record<string, string> = {
  upload: 'Uploading video',
  uploading: 'Uploading video',
  receiving: 'Receiving video',
  uploaded: 'Video uploaded',
  dispatching: 'Sending to worker',
  queued: 'Waiting for worker',
  starting: 'Starting analysis',
  court: 'Finding court geometry',
  tracking: 'Tracking players',
  pose: 'Estimating body pose',
  pose_and_racket: 'Tracking players, pose and racket',
  canonicalize_and_map: 'Mapping players to court',
  shuttle: 'Following the shuttle',
  phase: 'Recognising movement phases',
  render: 'Composing annotated video',
  encode: 'Preparing video for playback',
  complete: 'Analysis complete',
  failed: 'Analysis failed',
  dispatch_unknown: 'Checking submission',
};
const phaseStages = [
  'court',
  'pose_and_racket',
  'canonicalize_and_map',
  'shuttle',
  'phase',
  'render',
  'encode',
];
const modeStages: Record<AnalysisMode, string[]> = {
  combined: phaseStages,
  phases: phaseStages,
  court: ['court', 'render', 'encode'],
  tracking: ['tracking', 'canonicalize_and_map', 'render', 'encode'],
  pose: ['pose', 'canonicalize_and_map', 'render', 'encode'],
  racket: ['pose_and_racket', 'canonicalize_and_map', 'render', 'encode'],
  shuttle: ['shuttle', 'render', 'encode'],
};
const nav = [
  { id: '/', label: 'Overview', icon: Grid2X2 },
  { id: '/analyse', label: 'Analyse', icon: ScanLine },
  { id: '/rallies', label: 'My rallies', icon: Film },
  { id: '/compare', label: 'Compare', icon: GitCompareArrows },
];
function getRoute() {
  const [path, query = ''] = (window.location.hash.slice(1) || '/').split('?');
  return { path, params: new URLSearchParams(query) };
}
function go(path: string) {
  window.location.hash = path;
}
function safeRead(key: string) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function safeWrite(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* Optional preferences. */
  }
}

class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <div className="fatal">
        <Aperture size={42} />
        <h1>Let's get you back on court.</h1>
        <p>
          The workspace couldn't display this view. Your saved analysis links are still in this
          browser.
        </p>
        <button className="button primary" onClick={() => window.location.reload()}>
          Reload workspace
        </button>
      </div>
    ) : (
      this.props.children
    );
  }
}

export default function App() {
  return (
    <ErrorBoundary>
      <Workspace />
    </ErrorBoundary>
  );
}
function Workspace() {
  const [route, setRoute] = useState(getRoute);
  const [jobs, setJobs] = useState<StoredJob[]>(loadJobs);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [motionOff, setMotionOff] = useState(() => safeRead('rallymirror.motion') === 'off');
  const [systemReduced, setSystemReduced] = useState(
    () => matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  const [uploading, setUploading] = useState(false);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [uploadError, setUploadError] = useState('');
  const [retry, setRetry] = useState(0);
  const uploadController = useRef<AbortController | null>(null);
  const main = useRef<HTMLElement>(null);
  const reducedMotion = motionOff || systemReduced;
  const jobId = route.params.get('job');
  const selected = jobs.find((j) => j.id === jobId);
  const persist = useCallback((job: StoredJob) => setJobs(saveJob(job)), []);

  useEffect(() => {
    const change = () => {
      setRoute(getRoute());
      setMobileOpen(false);
      window.scrollTo({ top: 0, behavior: 'instant' });
    };
    window.addEventListener('hashchange', change);
    return () => window.removeEventListener('hashchange', change);
  }, []);
  useEffect(() => {
    const q = matchMedia('(prefers-reduced-motion: reduce)');
    const change = () => setSystemReduced(q.matches);
    q.addEventListener('change', change);
    return () => q.removeEventListener('change', change);
  }, []);
  useEffect(() => {
    document.documentElement.dataset.motion = reducedMotion ? 'off' : 'on';
  }, [reducedMotion]);
  useEffect(() => {
    const close = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMobileOpen(false);
    };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, []);
  useEffect(() => () => uploadController.current?.abort(), []);
  useEffect(() => {
    if (!selected || uploading) return;
    const controller = new AbortController();
    const initial = selected;
    api
      .watchJob(initial, {
        signal: controller.signal,
        onUpdate: (s) =>
          persist({
            ...initial,
            state: s.state,
            stage: s.stage,
            error: s.error,
            updatedAt: Date.now(),
          }),
      })
      .catch((e) => {
        if (!isAbortError(e)) setUploadError(errorMessage(e));
      });
    return () => controller.abort();
  }, [jobId, retry, uploading, persist]);

  async function submit(file: File, key: string, analysisMode: AnalysisMode) {
    if (uploading) return;
    setUploadError('');
    setUploading(true);
    setUploadPercent(0);
    const controller = new AbortController();
    uploadController.current = controller;
    let record: StoredJob | undefined;
    try {
      await videoMetadata(file);
      const created = await api.createJob(key, controller.signal, analysisMode);
      record = {
        id: created.id,
        analysis_mode: created.analysis_mode,
        token: created.token,
        name: file.name,
        size: file.size,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        state: 'uploading',
        stage: 'upload',
      };
      persist(record);
      go('/analyse?job=' + encodeURIComponent(record.id));
      await api.upload(created, file, { signal: controller.signal, onProgress: setUploadPercent });
      record = { ...record, state: 'uploaded', stage: 'uploaded', updatedAt: Date.now() };
      persist(record);
      const started = await api.startJob(record, controller.signal);
      record = { ...record, state: started.state, stage: started.state, updatedAt: Date.now() };
      persist(record);
      setRetry((v) => v + 1);
    } catch (e) {
      if (!isAbortError(e)) {
        const message = errorMessage(e);
        setUploadError(message);
        if (record) persist({ ...record, error: message, updatedAt: Date.now() });
      }
    } finally {
      setUploading(false);
      uploadController.current = null;
    }
  }

  const currentLabel =
    nav.find((n) => n.id === route.path)?.label ||
    (route.path === '/project' ? 'Project' : 'Workspace');
  return (
    <div className="app-shell" data-page={route.path}>
      <a
        className="skip-link"
        href="#main-content"
        onClick={(e) => {
          e.preventDefault();
          main.current?.focus();
        }}
      >
        Skip to content
      </a>
      {mobileOpen && (
        <button
          className="nav-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <aside className={`sidebar ${mobileOpen ? 'is-open' : ''}`} aria-label="Workspace navigation">
        <a className="brand" href="#/" aria-label="RallyMirror overview">
          <img src="/favicon.svg" alt="" />
          <span>
            RallyMirror<span className="brand-dot">.</span>
          </span>
        </a>
        <nav>
          {nav.map((n) => (
            <a
              key={n.id}
              href={'#' + n.id}
              className={'nav-item ' + (route.path === n.id ? 'active' : '')}
              aria-current={route.path === n.id ? 'page' : undefined}
            >
              <n.icon size={19} strokeWidth={1.7} />
              <span>{n.label}</span>
              {n.id === '/rallies' && jobs.length > 0 && (
                <span className="nav-count">{jobs.length}</span>
              )}
            </a>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <a href="#/project" className={'nav-item ' + (route.path === '/project' ? 'active' : '')}>
            <Layers3 size={18} />
            <span>Project & models</span>
          </a>
          <a
            href="https://github.com/felixxtng/RallyMirror"
            target="_blank"
            rel="noreferrer"
            className="nav-item"
          >
            <CircleHelp size={18} />
            <span>Documentation</span>
            <ExternalLink size={13} />
          </a>
          <div className="team">
            <span className="team-avatar">RM</span>
            <div>
              <strong>RallyMirror team</strong>
              <small>CVDL project workspace</small>
            </div>
          </div>
        </div>
      </aside>
      <div className="workspace-body">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              aria-expanded={mobileOpen}
              onClick={() => setMobileOpen(!mobileOpen)}
            >
              <Menu size={21} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={13} />
            <strong>{currentLabel}</strong>
          </div>
          <div className="topbar-actions">
            <button
              className="motion-button"
              aria-pressed={!reducedMotion}
              onClick={() => {
                const next = !motionOff;
                setMotionOff(next);
                safeWrite('rallymirror.motion', next ? 'off' : 'on');
              }}
              title={systemReduced ? 'System reduced motion is enabled' : undefined}
            >
              <Sparkles size={15} />
              <span>{reducedMotion ? 'Motion off' : 'Motion on'}</span>
            </button>
          </div>
        </header>
        <main ref={main} id="main-content" tabIndex={-1} className="main-content">
          <div key={route.path} className="page-enter">
            {route.path === '/' && <Overview reducedMotion={reducedMotion} jobs={jobs} />}
            {route.path === '/analyse' && (
              <>
                <PageHeading
                  title={
                    route.params.has('example')
                      ? 'Analysis examples'
                      : selected
                        ? 'Rally analysis'
                        : 'Analyse a rally'
                  }
                  description={
                    route.params.has('example')
                      ? 'Explore each feature or watch the combined result.'
                      : selected
                        ? selected.name
                        : 'Upload a raw MP4 to review player movement and court position.'
                  }
                />
                {uploadError && (
                  <div className="notice error" role="alert">
                    <CircleHelp size={18} />
                    <span>{uploadError}</span>
                    <button
                      className="icon-button"
                      aria-label="Dismiss error"
                      onClick={() => setUploadError('')}
                    >
                      <X size={16} />
                    </button>
                  </div>
                )}
                {route.params.has('example') ? (
                  <Review
                    key={'example-' + route.params.get('mode')}
                    example
                    exampleMode={
                      isAnalysisMode(route.params.get('mode'))
                        ? (route.params.get('mode') as AnalysisMode)
                        : 'combined'
                    }
                  />
                ) : selected ? (
                  <>
                    <JobProgress
                      job={selected}
                      uploading={uploading}
                      uploadPercent={uploadPercent}
                      onRefresh={() => {
                        setUploadError('');
                        setRetry((v) => v + 1);
                      }}
                    />
                    {selected.state === 'succeeded' && <Review key={selected.id} job={selected} />}
                    {!isActiveJob(selected.state) &&
                      selected.state !== 'succeeded' &&
                      !uploading && (
                        <button className="button secondary" onClick={() => go('/analyse')}>
                          Choose another clip
                        </button>
                      )}
                  </>
                ) : jobId ? (
                  <Empty
                    icon={<Film />}
                    title="This analysis isn't saved here."
                    text="Private analysis links belong to the browser that created them. Choose a new rally or open a saved analysis."
                    action="My rallies"
                    onAction={() => go('/rallies')}
                  />
                ) : (
                  <UploadPanel
                    busy={uploading}
                    onSubmit={submit}
                    initialMode={
                      isAnalysisMode(route.params.get('mode'))
                        ? (route.params.get('mode') as AnalysisMode)
                        : 'combined'
                    }
                  />
                )}
              </>
            )}
            {route.path === '/rallies' && (
              <Rallies jobs={jobs} onRemove={(id) => setJobs(removeJob(id))} />
            )}
            {route.path === '/compare' && <Compare />}
            {route.path === '/project' && <Project />}
            {!['/', '/analyse', '/rallies', '/compare', '/project'].includes(route.path) && (
              <Empty
                icon={<Focus />}
                title="A little outside the court."
                text="This page doesn't exist. Return to your workspace to continue."
                action="Back to overview"
                onAction={() => go('/')}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

function PageHeading({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <h1>{title}</h1>
        <p className="lede">{description}</p>
      </div>
      {children}
    </div>
  );
}

function Overview({ reducedMotion, jobs }: { reducedMotion: boolean; jobs: StoredJob[] }) {
  const [selectedModule, setSelectedModule] = useState(0);
  const storyRoot = useRef<HTMLElement>(null);
  const [scrollEnabled, setScrollEnabled] = useState(
    () => !reducedMotion && matchMedia('(min-width: 1100px)').matches,
  );
  const [cameraView, setCameraView] = useState(0);
  useEffect(() => {
    const media = matchMedia('(min-width: 1100px)');
    const update = () => setScrollEnabled(media.matches && !reducedMotion);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, [reducedMotion]);
  useEffect(() => {
    const root = storyRoot.current;
    if (!root || !scrollEnabled) {
      setCameraView(0);
      root?.style.setProperty('--story-progress', '0');
      return;
    }
    let frame = 0;
    const read = () => {
      frame = 0;
      const bounds = root.getBoundingClientRect();
      const progress = Math.min(
        1,
        Math.max(0, -bounds.top / Math.max(1, bounds.height - innerHeight)),
      );
      root.style.setProperty('--story-progress', String(progress));
      setCameraView(progress < 0.32 ? 0 : progress < 0.72 ? 1 : 2);
    };
    const queue = () => {
      if (!frame) frame = requestAnimationFrame(read);
    };
    window.addEventListener('scroll', queue, { passive: true });
    window.addEventListener('resize', queue);
    read();
    return () => {
      if (frame) cancelAnimationFrame(frame);
      window.removeEventListener('scroll', queue);
      window.removeEventListener('resize', queue);
    };
  }, [scrollEnabled]);
  function moveCamera(index: number) {
    const root = storyRoot.current;
    if (!root || !scrollEnabled) return;
    const top = root.getBoundingClientRect().top + scrollY;
    const distance = Math.max(1, root.offsetHeight - innerHeight);
    window.scrollTo({
      top: top + [0, 0.5, 1][index] * distance,
      behavior: reducedMotion ? 'instant' : 'smooth',
    });
  }
  const galleryItems = [
    {
      id: 'court-preview',
      title: 'Combined analysis',
      detail: 'All six features, on one actual rally.',
      image: POSTER,
      href: '#/analyse?example=1',
    },
    {
      id: 'court-data',
      title: 'Court dataset',
      detail: 'Source images and named keypoint annotations.',
      image: '/media/court-data.jpg',
      href: '#/project',
    },
    {
      id: 'clip-comparison',
      title: 'Clip comparison',
      detail: 'Two local videos with linked playback.',
      image: '/media/rally-comparison.jpg',
      href: '#/compare',
    },
    {
      id: 'analysis-pipeline',
      title: 'Rally analysis',
      detail: 'Court, players, pose, racket, shuttle and phases.',
      image: '/media/rally-source.jpg',
      href: '#/analyse',
    },
  ];
  const modules = [
    {
      name: 'Court',
      icon: Focus,
      label: 'Court mapping',
      detail:
        'Automatic court keypoints and floor mapping give player positions a consistent frame of reference.',
    },
    {
      name: 'Players',
      icon: ScanLine,
      label: 'Player tracking',
      detail: 'Player detection, tracking and near/far slots connect observations across a rally.',
    },
    {
      name: 'Pose',
      icon: Activity,
      label: 'Body pose',
      detail: 'Body landmarks reveal posture and movement throughout each sequence.',
    },
    {
      name: 'Racket',
      icon: Target,
      label: 'Racket localisation',
      detail: 'A dedicated model estimates the racket centre for the selected player.',
    },
    {
      name: 'Shuttle',
      icon: Aperture,
      label: 'Shuttle tracking',
      detail:
        'Temporal shuttle tracking builds a trajectory while retaining explicit gaps when it is missing.',
    },
    {
      name: 'Phases',
      icon: Layers3,
      label: 'Phase recognition',
      detail: 'Ready, movement, stroke, recovery and idle predictions add context over time.',
    },
  ];
  return (
    <div className="premium-overview">
      <section
        ref={storyRoot}
        className={`court-story ${scrollEnabled ? 'is-scroll-enabled' : ''}`}
        aria-label="RallyMirror court preview"
      >
        <div className="court-story__sticky">
          <section className="hero" aria-labelledby="hero-title">
            <div className="hero-grid" aria-hidden="true">
              <i />
              <i />
              <i />
              <i />
            </div>
            <div className="hero-copy">
              <h1 id="hero-title">
                <EditorialText text="Every rally." reducedMotion={reducedMotion} delay={80} />
                <br />
                <EditorialText text="A clearer" reducedMotion={reducedMotion} delay={180} />
                <br />
                <em>
                  <EditorialText text="picture." reducedMotion={reducedMotion} delay={280} />
                </em>
              </h1>
              <EditorialText
                as="p"
                text="Track players, map the court and review your rally."
                reducedMotion={reducedMotion}
                delay={400}
              />
              <div className="hero-actions">
                <a className="button primary lumen-adapted" href="#/analyse">
                  <Plus size={19} /> Analyse a rally
                </a>
                <a className="button ghost" href="#/analyse?example=1">
                  <Play size={16} /> Explore example
                </a>
              </div>
              {scrollEnabled && (
                <div className="camera-views" role="group" aria-label="Court camera views">
                  {['Court', 'Orbit', 'Shuttle'].map((label, index) => (
                    <button
                      key={label}
                      aria-pressed={cameraView === index}
                      onClick={() => moveCamera(index)}
                    >
                      <span className="camera-view-mark" aria-hidden="true" />
                      {label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="hero-visual">
              <Suspense
                fallback={
                  <div className="court-loading">
                    <Focus size={52} />
                    <span>Preparing the court</span>
                  </div>
                }
              >
                <CourtScene
                  reducedMotion={reducedMotion}
                  scrollRoot={scrollEnabled ? storyRoot : undefined}
                  className="court-scene"
                />
              </Suspense>
            </div>
          </section>
        </div>
      </section>
      <section className="module-section">
        <div className="section-heading">
          <div>
            <EditorialText as="h2" text="Vision modules" reducedMotion={reducedMotion} />
          </div>
          <a className="text-link" href="#/project">
            Inside the project <ArrowUpRight size={17} />
          </a>
        </div>
        <div className="module-tabs" role="tablist" aria-label="Vision modules">
          {modules.map((m, i) => (
            <button
              key={m.name}
              id={'module-tab-' + i}
              role="tab"
              aria-selected={selectedModule === i}
              aria-controls="module-panel"
              className={selectedModule === i ? 'active' : ''}
              onClick={() => setSelectedModule(i)}
              onKeyDown={(e) => {
                if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
                  e.preventDefault();
                  const n =
                    (i + (e.key === 'ArrowRight' ? 1 : -1) + modules.length) % modules.length;
                  setSelectedModule(n);
                  document.getElementById('module-tab-' + n)?.focus();
                }
              }}
            >
              <m.icon size={21} />
              <span>{m.name}</span>
            </button>
          ))}
        </div>
        <div
          id="module-panel"
          className="module-detail"
          role="tabpanel"
          aria-labelledby={'module-tab-' + selectedModule}
        >
          <strong>{modules[selectedModule].label}</strong>
          <p>{modules[selectedModule].detail}</p>
          <div className="module-actions">
            <a
              className="text-link"
              href={
                '#/analyse?example=1&mode=' +
                (['court', 'tracking', 'pose', 'racket', 'shuttle', 'phases'] as const)[
                  selectedModule
                ]
              }
            >
              View result <Play size={15} />
            </a>
            <a
              className="text-link"
              href={
                '#/analyse?mode=' +
                (['court', 'tracking', 'pose', 'racket', 'shuttle', 'phases'] as const)[
                  selectedModule
                ]
              }
            >
              Analyse a clip <ArrowUpRight size={15} />
            </a>
          </div>
        </div>
      </section>
      <section className="project-showcase" aria-labelledby="showcase-title">
        <EditorialReveal reducedMotion={reducedMotion} className="showcase-heading">
          <div>
            <h2 id="showcase-title">Inside RallyMirror</h2>
            <p>Explore the data, review a result or compare your clips.</p>
          </div>
        </EditorialReveal>
        <Suspense
          fallback={
            <div className="gallery-loading" role="status">
              Loading project examples…
            </div>
          }
        >
          <RallyGallery items={galleryItems} reducedMotion={reducedMotion} />
        </Suspense>
      </section>
      <section className="lower-grid">
        <div className="recent-block">
          <div className="section-heading">
            <h2>Recent analyses</h2>
            <a className="text-link" href="#/rallies">
              All rallies <ChevronRight size={16} />
            </a>
          </div>
          {jobs.length ? (
            <div className="recent-list">
              {jobs.slice(0, 3).map((j) => (
                <a
                  className="recent-row"
                  key={j.id}
                  href={'#/analyse?job=' + encodeURIComponent(j.id)}
                >
                  <span className="file-icon">
                    <Film size={21} />
                  </span>
                  <div>
                    <strong>{j.name}</strong>
                    <small>
                      {formatBytes(j.size)} · {new Date(j.createdAt).toLocaleDateString()}
                    </small>
                  </div>
                  <Status state={j.state} />
                  <ChevronRight size={17} />
                </a>
              ))}
            </div>
          ) : (
            <a className="example-card" href="#/analyse?example=1">
              <div className="example-thumb">
                <img
                  src={POSTER}
                  alt="Combined RallyMirror predictions on a badminton rally"
                  loading="lazy"
                />
                <span className="play-circle">
                  <Play size={18} fill="currentColor" />
                </span>
              </div>
              <div>
                <span className="eyebrow">PROJECT EXAMPLE</span>
                <h3>One rally. Six perspectives.</h3>
                <p>Watch the full pipeline, then explore each feature.</p>
                <span className="sample-label">ACTUAL MODEL OUTPUT</span>
              </div>
              <ArrowUpRight className="example-arrow" size={21} />
            </a>
          )}
        </div>
        <a className="compare-teaser" href="#/compare">
          <div className="compare-glyph" aria-hidden="true">
            <span>A</span>
            <span>B</span>
          </div>
          <div>
            <h2>Compare clips</h2>
            <p>Play two local clips together and adjust their timing.</p>
            <span className="text-link">
              Open comparison <ArrowUpRight size={18} />
            </span>
          </div>
        </a>
      </section>
      <section className="overview-cta">
        <EditorialText as="h2" text="Your next rally." reducedMotion={reducedMotion} />
        <a className="button primary" href="#/analyse">
          <Plus size={18} /> Analyse a rally
        </a>
      </section>
    </div>
  );
}

function Status({ state }: { state: string }) {
  return (
    <span className={'status status-' + state}>
      {state === 'succeeded' ? (
        <CheckCircle2 size={13} />
      ) : isActiveJob(state as StoredJob['state']) ? (
        <LoaderCircle className="spin" size={13} />
      ) : (
        <span className="status-dot" />
      )}
      {(
        {
          succeeded: 'Ready to review',
          failed: 'Needs attention',
          dispatch_unknown: 'Check submission',
          uploaded: 'Upload received',
          uploading: 'Uploading',
          receiving: 'Uploading',
          queued: 'Queued',
          running: 'Analysing',
          dispatching: 'Submitting',
        } as Record<string, string>
      )[state] || state}
    </span>
  );
}

function UploadPanel({
  busy,
  onSubmit,
  initialMode = 'combined',
}: {
  busy: boolean;
  onSubmit: (file: File, key: string, mode: AnalysisMode) => void;
  initialMode?: AnalysisMode;
}) {
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>(initialMode);
  const [file, setFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<{
    duration: number;
    width: number;
    height: number;
  } | null>(null);
  const [error, setError] = useState('');
  const [key, setKey] = useState('');
  const [drag, setDrag] = useState(false);
  const [checking, setChecking] = useState(false);
  const [connection, setConnection] = useState<'unknown' | 'online' | 'offline'>('unknown');
  const [connectionKey, setConnectionKey] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const selectionVersion = useRef(0);
  useEffect(() => {
    const c = new AbortController();
    api
      .health(c.signal)
      .then(() => setConnection('online'))
      .catch((e) => {
        if (!isAbortError(e)) setConnection('offline');
      });
    return () => c.abort();
  }, [connectionKey]);
  useEffect(
    () => () => {
      selectionVersion.current++;
    },
    [],
  );
  async function choose(candidate?: File) {
    if (!candidate) return;
    const version = ++selectionVersion.current;
    setFile(null);
    setMetadata(null);
    setError('');
    setChecking(true);
    try {
      const m = await videoMetadata(candidate);
      if (version === selectionVersion.current) {
        setFile(candidate);
        setMetadata(m);
      }
    } catch (e) {
      if (version === selectionVersion.current) setError(errorMessage(e));
    } finally {
      if (version === selectionVersion.current) setChecking(false);
    }
  }
  return (
    <div className="upload-layout">
      <section className="upload-card">
        <div className="section-heading">
          <h2>Start with your footage</h2>
          <span className="small-chip">RAW MP4</span>
        </div>
        <input
          ref={input}
          className="visually-hidden"
          id="rally-file"
          type="file"
          accept="video/mp4,.mp4"
          aria-label="Choose rally video"
          disabled={busy}
          onChange={(e) => {
            void choose(e.target.files?.[0]);
            e.target.value = '';
          }}
        />
        <button
          type="button"
          className={'dropzone ' + (drag ? 'dragging' : '')}
          disabled={busy || checking}
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            if (!busy && !checking) void choose(e.dataTransfer.files[0]);
          }}
        >
          <span className="upload-orbit">
            {checking ? (
              <LoaderCircle className="spin" size={28} />
            ) : file ? (
              <Check size={28} />
            ) : (
              <UploadCloud size={28} />
            )}
          </span>
          <strong>
            {checking ? 'Reading your video…' : file ? file.name : 'Drop your rally here'}
          </strong>
          <span>
            {file && metadata
              ? `${formatTime(metadata.duration)} · ${metadata.width} × ${metadata.height} · ${formatBytes(file.size)}`
              : 'or click to choose a video from your device'}
          </span>
          <small>MP4 · up to 60 seconds · 200 MiB</small>
        </button>
        {error && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (file && !checking) onSubmit(file, key, analysisMode);
          }}
        >
          <div className="analysis-picker">
            <label htmlFor="analysis-mode">Analysis mode</label>
            <select
              id="analysis-mode"
              value={analysisMode}
              disabled={busy}
              onChange={(e) => {
                if (isAnalysisMode(e.target.value)) setAnalysisMode(e.target.value);
              }}
            >
              {ANALYSIS_MODES.map((mode) => (
                <option key={mode} value={mode}>
                  {analysisInfo[mode].label}
                </option>
              ))}
            </select>
            <p>{analysisInfo[analysisMode].description}</p>
          </div>
          <details className="access-details">
            <summary>
              <ShieldCheck size={15} /> Team access <ChevronDown size={15} />
            </summary>
            <label htmlFor="access-key">
              Team access key <small>Optional for an unrestricted local server</small>
            </label>
            <input
              id="access-key"
              type="password"
              autoComplete="off"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="Enter only if your server requires it"
            />
            <p>This key stays in this session and is never saved in history.</p>
          </details>
          <button
            className="button primary upload-submit"
            disabled={!file || busy || checking}
            type="submit"
          >
            {busy ? <LoaderCircle className="spin" size={18} /> : <ScanLine size={18} />}
            {busy ? 'Preparing analysis…' : 'Analyse video'}
          </button>
        </form>
        <p className="privacy-note">
          <ShieldCheck size={14} /> Uploads run inference only. They do not train or change the
          models.
        </p>
      </section>
      <aside className="upload-info">
        <div className="connection-card">
          <span className={'connection-icon ' + connection}>
            {connection === 'online' ? <Wifi size={21} /> : <WifiOff size={21} />}
          </span>
          <h3>
            {connection === 'online'
              ? 'Analysis service connected'
              : connection === 'offline'
                ? 'Connect your analysis service'
                : 'Checking analysis service'}
          </h3>
          <p>
            {connection === 'offline'
              ? 'The interface is ready. Start the RallyMirror API and worker to process a new rally.'
              : 'Choose one feature or combine all six. Your clip returns as an annotated video with a run report.'}
          </p>
          {connection === 'offline' && (
            <button
              className="button secondary compact"
              onClick={() => setConnectionKey((k) => k + 1)}
            >
              Check connection
            </button>
          )}
        </div>
        <h3>A good clip makes a difference.</h3>
        <ul className="check-list">
          <li>
            <Check size={15} /> Keep the whole court in view.
          </li>
          <li>
            <Check size={15} /> Use a steady camera and a single rally.
          </li>
          <li>
            <Check size={15} /> Around 30 fps is best for phase review.
          </li>
          <li>
            <Check size={15} /> Upload the raw clip without overlays.
          </li>
        </ul>
        <div className="info-note">
          <VolumeX size={17} />
          <p>Output is silent. Other resolutions are fitted to the model's 1280 × 720 canvas.</p>
        </div>
        <a className="text-link" href="#/analyse?example=1">
          <Play size={16} /> Explore the recorded example
        </a>
      </aside>
    </div>
  );
}

function JobProgress({
  job,
  uploading,
  uploadPercent,
  onRefresh,
}: {
  job: StoredJob;
  uploading: boolean;
  uploadPercent: number;
  onRefresh: () => void;
}) {
  if (job.state === 'succeeded')
    return (
      <div className="complete-banner">
        <CheckCircle2 size={20} />
        <span>Your analysis is ready.</span>
        <Status state={job.state} />
      </div>
    );
  const stages = modeStages[job.analysis_mode ?? 'combined'];
  const activeIndex = stages.indexOf(job.stage);
  return (
    <section className="progress-card" aria-live="polite">
      <div className="section-heading">
        <div>
          <p className="eyebrow">{uploading ? 'UPLOADING YOUR CLIP' : 'ANALYSIS STATUS'}</p>
          <h2>
            {uploading
              ? `Uploading ${uploadPercent}%`
              : stageNames[job.stage] || 'Checking analysis status'}
          </h2>
        </div>
        <Status state={job.state} />
      </div>
      {uploading ? (
        <div
          className="progress-track"
          role="progressbar"
          aria-label="Video upload"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={uploadPercent}
        >
          <span style={{ width: uploadPercent + '%' }} />
        </div>
      ) : (
        <div
          className="stage-track"
          style={{ gridTemplateColumns: `repeat(${stages.length}, minmax(0, 1fr))` }}
        >
          {stages.map((s, i) => (
            <div key={s} className={i < activeIndex ? 'done' : i === activeIndex ? 'current' : ''}>
              <span>{i < activeIndex ? <Check size={13} /> : i + 1}</span>
              <small>
                {
                  (
                    {
                      court: 'Court',
                      tracking: 'Players',
                      pose: 'Pose',
                      pose_and_racket: 'Players',
                      canonicalize_and_map: ['tracking', 'pose', 'racket'].includes(
                        job.analysis_mode ?? '',
                      )
                        ? 'Identity'
                        : 'Mapping',
                      shuttle: 'Shuttle',
                      phase: 'Phases',
                      render: 'Render',
                      encode: 'Encode',
                    } as Record<string, string>
                  )[s]
                }
              </small>
            </div>
          ))}
        </div>
      )}
      {job.error && (
        <div className="notice error">
          <CircleHelp size={17} />
          <span>{job.error}</span>
        </div>
      )}
      <p className="muted">
        {job.state === 'dispatch_unknown'
          ? 'Submission is uncertain. Check the existing job before submitting again to avoid duplicate processing.'
          : isActiveJob(job.state)
            ? 'You can leave this view and reopen the analysis from My rallies. Processing time depends on the clip and available hardware.'
            : uploading
              ? 'Keep this page open until your upload finishes.'
              : 'Check the existing job status before submitting the video again.'}
      </p>
      {!uploading && (
        <button className="button secondary compact" onClick={onRefresh}>
          Refresh status
        </button>
      )}
    </section>
  );
}

function Review({
  example = false,
  job,
  exampleMode = 'combined',
}: {
  example?: boolean;
  job?: StoredJob;
  exampleMode?: AnalysisMode;
}) {
  const mode = example ? exampleMode : (job?.analysis_mode ?? 'combined');
  const info = analysisInfo[mode];
  const [links, setLinks] = useState<ResultLinks | null>(null);
  const [report, setReport] = useState<RunReport | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(!example);
  const [refresh, setRefresh] = useState(0);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(1);
  const [tab, setTab] = useState('overview');
  const video = useRef<HTMLVideoElement>(null);
  const restoreTime = useRef(0);
  const lastRefresh = useRef(0);
  const mediaFailures = useRef(0);
  useEffect(() => {
    if (!example) return;
    const c = new AbortController();
    fetch(exampleReport(mode), { signal: c.signal })
      .then((response) => {
        if (!response.ok) throw new Error('Report unavailable');
        return response.json() as Promise<RunReport>;
      })
      .then((next) => {
        if (!c.signal.aborted) setReport(next);
      })
      .catch(() => {
        /* The recorded video remains reviewable if its report is offline. */
      });
    return () => c.abort();
  }, [example, mode]);
  useEffect(() => {
    if (example || !job) return;
    const c = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    setError('');
    setLoading(true);
    api
      .getResultLinks(job, c.signal)
      .then(async (next) => {
        if (c.signal.aborted) return;
        restoreTime.current = video.current?.currentTime || 0;
        setLinks(next);
        setLoading(false);
        timer = setTimeout(
          () => setRefresh((v) => v + 1),
          Math.max(10000, next.expiresAt - Date.now()),
        );
        try {
          const r = await api.getReport(next.reportUrl, c.signal);
          if (!c.signal.aborted) setReport(r);
        } catch (e) {
          if (!isAbortError(e))
            setError(
              'The video is ready, but the run report could not be loaded. Refresh to try again.',
            );
        }
      })
      .catch((e) => {
        if (!isAbortError(e)) {
          setError(errorMessage(e));
          setLoading(false);
        }
      });
    return () => {
      c.abort();
      clearTimeout(timer);
    };
  }, [job?.id, example, refresh]);
  const source = example ? exampleVideo(mode) : links?.videoUrl;
  const fps = finiteNumber(report?.input?.fps) ? report.input.fps : 30;
  function step(direction: number) {
    if (video.current) {
      video.current.pause();
      video.current.currentTime = Math.min(
        duration,
        Math.max(0, video.current.currentTime + direction / fps),
      );
    }
  }
  function refreshMedia() {
    if (example || Date.now() - lastRefresh.current < 5000) return;
    if (mediaFailures.current >= 2) {
      setError(
        'The video could not be played after refreshing its link. Try downloading it or refresh the results manually.',
      );
      return;
    }
    mediaFailures.current++;
    lastRefresh.current = Date.now();
    restoreTime.current = video.current?.currentTime || 0;
    setRefresh((v) => v + 1);
  }
  const warnings = Array.isArray(report?.warnings)
    ? report.warnings.filter((w) => typeof w === 'string')
    : [];
  const layerModes: Record<string, Exclude<AnalysisMode, 'combined'>> = {
    'Court geometry': 'court',
    'Player tracking': 'tracking',
    'Body pose': 'pose',
    'Player-owned racket': 'racket',
    'Shuttle trajectory': 'shuttle',
    'Predicted play phases': 'phases',
  };
  const unavailableLayer = (name: string) =>
    report?.modules?.[layerModes[name]]?.status === 'unavailable' ||
    warnings.some(
      (w) =>
        (name.includes('Court') && /court.*(reject|unavailable)/i.test(w)) ||
        (/No player poses/i.test(w) && /Player|Predicted|Body pose/.test(name)),
    );
  const warningBanner = warnings.length > 0 && (
    <div className="notice warning" role="status">
      <CircleHelp size={18} />
      <span>
        <strong>Review with care.</strong> {warnings[0]}
        {warnings.length > 1 ? ` (${warnings.length - 1} more in Quality & limitations.)` : ''}
      </span>
    </div>
  );
  return (
    <section className="review-layout">
      {example && (
        <>
          <div className="notice example-notice">
            <Eye size={18} />
            <span>
              <strong>
                Recorded{' '}
                {mode === 'combined' ? 'combined analysis' : analysisInfo[mode].label.toLowerCase()}
                .
              </strong>{' '}
              Actual model predictions from one raw rally. Explore each feature or the combined
              view.
            </span>
          </div>
          <nav className="example-modes" aria-label="Example analysis mode">
            {ANALYSIS_MODES.map((item) => (
              <a
                key={item}
                className={mode === item ? 'active' : ''}
                aria-current={mode === item ? 'page' : undefined}
                href={'#/analyse?example=1&mode=' + item}
              >
                {item === 'combined' ? 'All features' : analysisInfo[item].label}
              </a>
            ))}
          </nav>
        </>
      )}
      {error && (
        <div className="notice error" role="alert">
          <CircleHelp size={18} />
          <span>{error}</span>
          <button className="button secondary compact" onClick={() => setRefresh((v) => v + 1)}>
            Refresh results
          </button>
        </div>
      )}
      {warningBanner}
      <div className="review-grid">
        <div className="video-card">
          <div className="video-top">
            <span>
              <span className="record-dot" />{' '}
              {mode === 'combined' ? 'COMBINED ANALYSIS' : info.label.toUpperCase()}
            </span>
            <span>
              <VolumeX size={13} /> SILENT OUTPUT
            </span>
          </div>
          <div className="video-stage">
            {source ? (
              <video
                ref={video}
                key={example ? mode : job?.id}
                src={source}
                poster={example ? examplePoster(mode) : undefined}
                controls
                playsInline
                preload="metadata"
                aria-label={
                  example
                    ? 'Recorded ' + info.label.toLowerCase() + ' example'
                    : 'Annotated rally video'
                }
                onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
                onLoadedMetadata={(e) => {
                  const v = e.currentTarget;
                  if (finiteNumber(v.duration)) setDuration(v.duration);
                  v.playbackRate = rate;
                  if (restoreTime.current) {
                    v.currentTime = Math.min(restoreTime.current, v.duration);
                    restoreTime.current = 0;
                  }
                }}
                onError={() => {
                  if (example)
                    setError('The recorded example could not be played in this browser.');
                  else refreshMedia();
                }}
              />
            ) : (
              <div className="video-loading">
                {loading ? <LoaderCircle size={30} className="spin" /> : <Film size={36} />}
                <span>{loading ? 'Preparing your video…' : 'Result not available'}</span>
              </div>
            )}
          </div>
          <div className="playback-tools">
            <div className="frame-tools">
              <button
                className="icon-button"
                aria-label="Previous frame"
                disabled={!source}
                onClick={() => step(-1)}
              >
                <StepBack size={18} />
              </button>
              <button
                className="icon-button"
                aria-label="Next frame"
                disabled={!source}
                onClick={() => step(1)}
              >
                <StepForward size={18} />
              </button>
              <span className="timecode">
                {formatTime(time)} <span>/ {formatTime(duration)}</span>
              </span>
            </div>
            <label className="rate-select">
              Speed
              <select
                aria-label="Playback speed"
                value={rate}
                onChange={(e) => {
                  const r = Number(e.target.value);
                  setRate(r);
                  if (video.current) video.current.playbackRate = r;
                }}
              >
                {[0.25, 0.5, 1, 1.5, 2].map((r) => (
                  <option key={r} value={r}>
                    {r}×
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>
        <aside className="review-aside">
          <h2>{info.title}</h2>
          <p>{info.description}</p>
          <div className="layer-list">
            {info.layers.map((t) => (
              <div key={t} className={unavailableLayer(t) ? 'unavailable-layer' : ''}>
                {unavailableLayer(t) ? <CircleHelp size={16} /> : <Layers3 size={16} />}
                <span>
                  {t}
                  {unavailableLayer(t) ? ' — unavailable' : ''}
                </span>
              </div>
            ))}
          </div>
          <p className="caption">
            Overlays are rendered into the video. Predictions can be inaccurate or unavailable;
            check the run warnings.
          </p>
          {source && (
            <a className="button primary" href={source} download={'rallymirror-' + mode + '.mp4'}>
              <Download size={16} /> Download video
            </a>
          )}
          {links && (
            <a className="button secondary" href={links.reportUrl} target="_blank" rel="noreferrer">
              <FileJson size={16} /> Open run report
            </a>
          )}
          {example && (
            <a
              className="button secondary"
              href={exampleReport(mode)}
              target="_blank"
              rel="noreferrer"
            >
              <FileJson size={16} /> Open run report
            </a>
          )}
          {example && (
            <a className="text-link example-run-link" href={'#/analyse?mode=' + mode}>
              Use your own video <ArrowUpRight size={16} />
            </a>
          )}
        </aside>
      </div>
      {report?.modules && <PredictionCoverage report={report} />}
      <div className="review-details">
        <div className="detail-tabs" role="tablist" aria-label="Analysis details">
          {['overview', 'quality', 'technical'].map((t) => (
            <button
              id={'review-' + t}
              key={t}
              role="tab"
              aria-controls="review-detail-panel"
              aria-selected={tab === t}
              className={tab === t ? 'active' : ''}
              onClick={() => setTab(t)}
            >
              {
                (
                  {
                    overview: 'Overview',
                    quality: 'Quality & limitations',
                    technical: 'Run details',
                  } as Record<string, string>
                )[t]
              }
            </button>
          ))}
        </div>
        <div id="review-detail-panel" role="tabpanel" aria-labelledby={'review-' + tab}>
          {tab === 'overview' && (
            <div className="detail-content">
              <Focus size={29} />
              <div>
                <h3>Review the evidence in context.</h3>
                <p>
                  {example
                    ? 'All seven example views share the same inference results. The six focused videos show individual layers; the combined video overlays them together. No manual labels are used to draw these predictions.'
                    : 'Use slow playback and frame stepping to inspect the output. The run report records module coverage, warnings, model checksums and processing times.'}
                </p>
              </div>
            </div>
          )}
          {tab === 'quality' && (
            <div className="quality-content">
              {warnings.length ? (
                warnings.map((w, i) => (
                  <div className="notice warning" key={i}>
                    <CircleHelp size={17} />
                    <span>{w}</span>
                  </div>
                ))
              ) : (
                <p>
                  {report
                    ? 'The run report contains no warnings. This does not guarantee that every prediction is accurate.'
                    : 'Quality information will appear when the run report is available.'}
                </p>
              )}
              <p className="muted">
                Missing points mean unknown, not zero. Phase labels describe stages of play, not
                specific stroke types. Court coordinates apply to the floor plane.
              </p>
            </div>
          )}
          {tab === 'technical' && (
            <div className="technical-content">
              {report ? (
                <>
                  <dl className="report-stats">
                    <div>
                      <dt>Processing time</dt>
                      <dd>
                        {finiteNumber(report.total_seconds)
                          ? report.total_seconds.toFixed(1) + ' s'
                          : 'Unavailable'}
                      </dd>
                    </div>
                    <div>
                      <dt>Compute device</dt>
                      <dd>{typeof report.device === 'string' ? report.device : 'Unavailable'}</dd>
                    </div>
                    <div>
                      <dt>Input frame rate</dt>
                      <dd>
                        {finiteNumber(report.input?.fps)
                          ? report.input.fps.toFixed(1) + ' fps'
                          : 'Unavailable'}
                      </dd>
                    </div>
                    <div>
                      <dt>Player observations</dt>
                      <dd>
                        {finiteNumber(report.pose_records)
                          ? report.pose_records.toLocaleString()
                          : 'Unavailable'}
                      </dd>
                    </div>
                  </dl>
                  {report.stage_seconds && (
                    <div className="stage-times">
                      {Object.entries(report.stage_seconds)
                        .filter(([, seconds]) => finiteNumber(seconds))
                        .map(([stage, seconds]) => (
                          <div key={stage}>
                            <span>{stageNames[stage] || stage}</span>
                            <strong>{seconds.toFixed(1)} s</strong>
                          </div>
                        ))}
                    </div>
                  )}
                  <p className="muted">
                    Processing time excludes upload and queue delay. These are run diagnostics, not
                    model accuracy scores.
                  </p>
                </>
              ) : (
                <p>The run report is not available.</p>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function PredictionCoverage({ report }: { report: RunReport }) {
  const rows = ANALYSIS_MODES.flatMap((mode) => {
    if (mode === 'combined') return [];
    const data = report.modules?.[mode];
    if (!data || data.status === 'not_requested' || !finiteNumber(data.records)) return [];
    return [{ mode, ...data }];
  });
  if (!rows.length) return null;
  return (
    <section className="prediction-coverage" aria-label="Prediction coverage">
      <div className="section-heading">
        <h2>Prediction coverage</h2>
        <span className="small-chip">THIS RUN</span>
      </div>
      <dl className="coverage-grid">
        {rows.map((row) => (
          <div key={row.mode}>
            <dt>{analysisInfo[row.mode].label}</dt>
            <dd>
              {row.status === 'unavailable'
                ? 'Unavailable'
                : row.mode === 'court'
                  ? 'Calibrated'
                  : row.records.toLocaleString()}
            </dd>
            <span>{row.mode === 'court' ? 'Floor geometry' : 'Available predictions'}</span>
          </div>
        ))}
      </dl>
      <p>
        Counts show available predictions, not accuracy.{' '}
        {report.inference_mode === 'combined'
          ? 'The example views share one combined inference run.'
          : ''}
      </p>
    </section>
  );
}

function Rallies({ jobs, onRemove }: { jobs: StoredJob[]; onRemove: (id: string) => void }) {
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const visible = jobs.filter(
    (j) =>
      j.name.toLowerCase().includes(search.toLowerCase()) &&
      (filter === 'all' || (filter === 'ready' ? j.state === 'succeeded' : isActiveJob(j.state))),
  );
  return (
    <>
      <PageHeading title="My rallies" description="Revisit the analyses saved in this browser.">
        <a href="#/analyse" className="button primary">
          <Plus size={18} /> Analyse a rally
        </a>
      </PageHeading>
      <div className="library-toolbar">
        <input
          aria-label="Search rallies"
          placeholder="Find a rally by filename…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="segmented" aria-label="Filter rallies">
          {['all', 'ready', 'processing'].map((f) => (
            <button
              key={f}
              aria-pressed={filter === f}
              className={filter === f ? 'active' : ''}
              onClick={() => setFilter(f)}
            >
              {f === 'all' ? 'All rallies' : f === 'ready' ? 'Ready' : 'Processing'}
            </button>
          ))}
        </div>
      </div>
      {visible.length ? (
        <div className="rally-list">
          {visible.map((j) => (
            <article className="rally-row" key={j.id}>
              <span className="file-icon">
                <Film size={24} />
              </span>
              <div className="rally-name">
                <a href={'#/analyse?job=' + encodeURIComponent(j.id)}>{j.name}</a>
                <small>
                  {analysisInfo[j.analysis_mode ?? 'combined'].label} · {formatBytes(j.size)} ·{' '}
                  {new Date(j.createdAt).toLocaleString()}
                </small>
              </div>
              <Status state={j.state} />
              <a
                className="button secondary compact"
                href={'#/analyse?job=' + encodeURIComponent(j.id)}
              >
                Open <ChevronRight size={15} />
              </a>
              <button
                className="icon-button"
                aria-label={'Remove ' + j.name + ' from browser history'}
                onClick={() => {
                  if (
                    window.confirm(
                      'Remove this link from browser history? This does not delete the server video.',
                    )
                  )
                    onRemove(j.id);
                }}
              >
                <Trash2 size={17} />
              </button>
            </article>
          ))}
        </div>
      ) : (
        <Empty
          icon={<Film size={34} />}
          title={jobs.length ? 'No matching rallies.' : 'Your next rally starts here.'}
          text={
            jobs.length
              ? 'Try another filename or filter.'
              : 'Upload a raw clip to start your collection, or explore the recorded combined analysis.'
          }
          action={jobs.length ? 'Clear filters' : 'Analyse a rally'}
          onAction={() => (jobs.length ? (setSearch(''), setFilter('all')) : go('/analyse'))}
        />
      )}
      <p className="library-privacy">
        <ShieldCheck size={15} /> History stores private result access tokens in this browser.
        Removing a history entry does not delete server-side files.
      </p>
    </>
  );
}

function Compare() {
  const [clips, setClips] = useState<[string | null, string | null]>([null, null]);
  const [names, setNames] = useState(['Attempt A', 'Attempt B']);
  const [offset, setOffset] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState('');
  const refs = [useRef<HTMLVideoElement>(null), useRef<HTMLVideoElement>(null)];
  const urls = useRef<string[]>([]);
  const comparisonMounted = useRef(true);
  const comparisonVersion = useRef([0, 0]);
  useEffect(() => {
    comparisonMounted.current = true;
    return () => {
      comparisonMounted.current = false;
      comparisonVersion.current = comparisonVersion.current.map((v) => v + 1);
      urls.current.forEach((u) => URL.revokeObjectURL(u));
    };
  }, []);
  useEffect(() => {
    const a = refs[0].current;
    const b = refs[1].current;
    if (!a || !b || !clips[0] || !clips[1]) return;
    const sync = () => {
      if (!Number.isFinite(b.duration) || b.duration <= 0) return;
      const relativeTime = a.currentTime + offset;
      const target = Math.min(b.duration, Math.max(0, relativeTime));
      if (Math.abs(b.currentTime - target) > 0.1) b.currentTime = target;
      b.playbackRate = a.playbackRate;
      // A negative offset means B waits at its first frame until its start.
      // Do not repeatedly play and rewind it during that waiting interval.
      if (a.paused || relativeTime < 0) {
        b.pause();
        return;
      }
      if (relativeTime >= b.duration) {
        a.pause();
        b.pause();
        setPlaying(false);
        return;
      }
      if (b.paused) {
        void b.play().catch((e) => {
          if (isAbortError(e) || !comparisonMounted.current) return;
          a.pause();
          setPlaying(false);
          setError('The second clip could not start. Try playback again.');
        });
      }
    };
    const pause = () => {
      b.pause();
      setPlaying(false);
    };
    const play = () => {
      sync();
      setPlaying(!a.paused);
    };
    a.addEventListener('timeupdate', sync);
    a.addEventListener('seeking', sync);
    a.addEventListener('play', play);
    a.addEventListener('pause', pause);
    a.addEventListener('ended', pause);
    a.addEventListener('ratechange', sync);
    b.addEventListener('loadedmetadata', sync);
    sync();
    return () => {
      a.removeEventListener('timeupdate', sync);
      a.removeEventListener('seeking', sync);
      a.removeEventListener('play', play);
      a.removeEventListener('pause', pause);
      a.removeEventListener('ended', pause);
      a.removeEventListener('ratechange', sync);
      b.removeEventListener('loadedmetadata', sync);
    };
  }, [clips, offset]);
  async function choose(file: File | undefined, i: number) {
    if (!file) return;
    const version = ++comparisonVersion.current[i];
    setError('');
    try {
      await videoMetadata(file);
      if (!comparisonMounted.current || comparisonVersion.current[i] !== version) return;
      refs.forEach((r) => r.current?.pause());
      setPlaying(false);
      const url = URL.createObjectURL(file);
      if (urls.current[i]) URL.revokeObjectURL(urls.current[i]);
      urls.current[i] = url;
      setClips((c) => {
        const next: [string | null, string | null] = [...c];
        next[i] = url;
        return next;
      });
      setNames((n) => n.map((v, k) => (k === i ? file.name : v)));
    } catch (e) {
      if (comparisonMounted.current && comparisonVersion.current[i] === version)
        setError(errorMessage(e));
    }
  }
  async function toggle() {
    const a = refs[0].current;
    if (!a) return;
    if (a.paused) {
      try {
        await a.play();
      } catch {
        setError('Your browser could not play this video.');
      }
    } else a.pause();
  }
  function loadExamplePair() {
    refs.forEach((ref) => ref.current?.pause());
    comparisonVersion.current = comparisonVersion.current.map((value) => value + 1);
    urls.current.forEach((url) => URL.revokeObjectURL(url));
    urls.current = [];
    setClips(['/media/source-analysis.mp4', exampleVideo('combined')]);
    setNames(['Source rally', 'Combined analysis']);
    setOffset(0);
    setPlaying(false);
    setError('');
  }
  return (
    <>
      <PageHeading
        title="Compare clips"
        description="Compare two local clips with linked playback and adjustable timing."
      >
        <button className="button secondary" onClick={loadExamplePair}>
          <Play size={16} /> Load project example
        </button>
      </PageHeading>
      <div className="notice">
        <ShieldCheck size={18} />
        <span>
          Local comparison. These files stay on your device and are not uploaded or analysed.
        </span>
      </div>
      {error && (
        <div className="notice error" role="alert">
          {error}
        </div>
      )}
      <div className="compare-grid">
        {[0, 1].map((i) => (
          <section className="compare-video" key={i}>
            <div className="section-heading">
              <span className="attempt-badge">{i === 0 ? 'A' : 'B'}</span>
              <h3>{names[i]}</h3>
              <label className="button secondary compact" htmlFor={'compare-' + i}>
                <Plus size={14} /> Choose clip
              </label>
              <input
                id={'compare-' + i}
                className="visually-hidden"
                type="file"
                accept="video/mp4,.mp4"
                aria-label={'Choose attempt ' + (i === 0 ? 'A' : 'B')}
                onChange={(e) => {
                  void choose(e.target.files?.[0], i);
                  e.target.value = '';
                }}
              />
            </div>
            {clips[i] ? (
              <video
                ref={refs[i]}
                src={clips[i]!}
                controls={i === 0}
                muted
                playsInline
                preload="metadata"
                aria-label={'Attempt ' + (i === 0 ? 'A' : 'B') + ' video'}
                onError={() => setError('One of these clips could not be decoded. Try H.264 MP4.')}
              />
            ) : (
              <label className="compare-empty" htmlFor={'compare-' + i}>
                <GitCompareArrows size={35} />
                <strong>Add attempt {i === 0 ? 'A' : 'B'}</strong>
                <span>MP4 · up to 60 seconds</span>
              </label>
            )}
          </section>
        ))}
      </div>
      <section className="compare-controls">
        <button
          className="button primary"
          disabled={!clips[0] || !clips[1]}
          onClick={() => void toggle()}
        >
          {playing ? <Pause size={17} /> : <Play size={17} />}
          {playing ? 'Pause both' : 'Play both'}
        </button>
        <label htmlFor="alignment">
          Attempt B offset{' '}
          <span>
            {offset > 0 ? '+' : ''}
            {offset.toFixed(1)} s
          </span>
        </label>
        <input
          id="alignment"
          type="range"
          min="-10"
          max="10"
          step="0.1"
          value={offset}
          onChange={(e) => setOffset(Number(e.target.value))}
        />
        <button className="button secondary compact" onClick={() => setOffset(0)}>
          Reset alignment
        </button>
      </section>
      <div className="comparison-note">
        <Focus size={25} />
        <div>
          <h3>Align a moment that matters.</h3>
          <p>
            Use the offset to align the beginning of a movement or swing. This is a playback
            comparison; automatic ghost alignment and cross-camera performance scoring are not
            implemented.
          </p>
        </div>
      </div>
    </>
  );
}

function Project() {
  const models = [
    {
      name: 'Player tracking',
      method: 'YOLOv11n + ByteTrack',
      text: 'Detect people, track IDs and organise the two players into near/far slots.',
    },
    {
      name: 'Body pose',
      method: 'MediaPipe Pose Landmarker',
      text: 'A frozen body model supplies keypoints for per-player movement features.',
    },
    {
      name: 'Racket centre',
      method: 'Pose-conditioned heatmap model',
      text: 'Selected v006 model: 3,259 training frames from 25 rallies.',
    },
    {
      name: 'Shuttle localisation',
      method: 'Three-frame encoder–decoder',
      text: 'Heatmap predictions and offline temporal refinement retain missing-detection states.',
    },
    {
      name: 'Court geometry',
      method: 'U-Net-style keypoint CNN',
      text: 'Trained from scratch. Predicted floor points support calibration and homography.',
    },
    {
      name: 'Movement phases',
      method: 'Causal temporal convolutional network',
      text: 'Pose, velocity, court and shuttle context produce five phase classes.',
    },
  ];
  return (
    <>
      <PageHeading
        title="Project overview"
        description="Six specialist modules turn rally footage into coordinated visual evidence."
      />
      <section className="project-intro">
        <div>
          <h2>Badminton video analysis</h2>
        </div>
        <p>
          Manual video review makes it difficult to connect where a player moved, how they moved and
          what happened next. RallyMirror brings player, racket, shuttle and court observations into
          one review workflow.
        </p>
      </section>
      <div className="section-heading">
        <h2>Inside the pipeline</h2>
        <a
          className="text-link"
          href="https://github.com/felixxtng/RallyMirror"
          target="_blank"
          rel="noreferrer"
        >
          View source <ExternalLink size={15} />
        </a>
      </div>
      <div className="model-table">
        {models.map((m, i) => (
          <article key={m.name}>
            <span className="model-index">0{i + 1}</span>
            <h3>{m.name}</h3>
            <div>
              <strong>{m.method}</strong>
              <p>{m.text}</p>
            </div>
          </article>
        ))}
      </div>
      <section className="evidence-section">
        <div className="section-heading">
          <div>
            <h2>Module evidence</h2>
          </div>
          <span className="small-chip">NOT OVERALL SYSTEM ACCURACY</span>
        </div>
        <div className="evidence-grid">
          <article>
            <span className="eyebrow">RACKET V006 / VALIDATION</span>
            <div className="metric-pair">
              <div>
                <strong>
                  88.1<span>%</span>
                </strong>
                <span>Near-side PCK20</span>
              </div>
              <div>
                <strong>
                  84.0<span>%</span>
                </strong>
                <span>Far-side PCK20</span>
              </div>
            </div>
            <p>
              444 labelled validation frames, 888 queries. PCK20 counts predictions within 20
              original-video pixels; missing predictions count as failures.
            </p>
          </article>
          <article>
            <span className="eyebrow">SHUTTLE / EXPLORATORY STUDY</span>
            <div className="metric-pair">
              <div>
                <strong>
                  71.17<span>%</span>
                </strong>
                <span>Original V2 micro F1</span>
              </div>
              <div>
                <strong>
                  78.06<span>%</span>
                </strong>
                <span>Offline V2 micro F1</span>
              </div>
            </div>
            <p>
              Four exploratory rallies using frozen V2. This is not the selected combined-pipeline
              benchmark; independent-match performance is unverified.
            </p>
          </article>
        </div>
      </section>
      <section className="dataset-section">
        <div>
          <h2>Dataset annotations</h2>
          <p>
            Source media and annotations belong together. Separate training, validation and test
            groups help measure performance beyond the footage used to fit a model.
          </p>
        </div>
        <dl>
          <div>
            <dt>Court</dt>
            <dd>Named keypoint clicks and visibility · JSON</dd>
          </div>
          <div>
            <dt>Shuttle</dt>
            <dd>Frame, centre and visibility · CSV</dd>
          </div>
          <div>
            <dt>Racket</dt>
            <dd>Player-owned centre and unknown masks · JSON</dd>
          </div>
          <div>
            <dt>Phases</dt>
            <dd>Per-player frame intervals · JSONL</dd>
          </div>
        </dl>
      </section>
      <div className="notice warning">
        <CircleHelp size={19} />
        <span>
          Predictions need human review. Court calibration can fail; tracking can switch identities;
          phases do not classify individual stroke types. Camera angle, occlusion and motion blur
          can reduce reliability. Review the footage alongside any predicted measurements.
        </span>
      </div>
      <p className="credits">
        3D interaction adapted from{' '}
        <a href="https://github.com/MengTo/threeui" target="_blank" rel="noreferrer">
          ThreeUI Community by Meng To
        </a>
        . MIT source attribution and bundled font licences are included with this project.
      </p>
    </>
  );
}

function Empty({
  icon,
  title,
  text,
  action,
  onAction,
}: {
  icon: ReactNode;
  title: string;
  text: string;
  action: string;
  onAction: () => void;
}) {
  return (
    <div className="empty">
      <span className="empty-icon">{icon}</span>
      <h2>{title}</h2>
      <p>{text}</p>
      <button className="button primary" onClick={onAction}>
        {action}
      </button>
    </div>
  );
}

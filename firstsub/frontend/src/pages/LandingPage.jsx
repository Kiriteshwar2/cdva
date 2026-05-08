import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Atom, Boxes, FlaskConical, Sparkles } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="landing-page">
      {/* NAVBAR */}
      <header className="landing-navbar">
        <div>
          <h2>CDVAE Lab</h2>
          <p>Materials AI Platform</p>
        </div>

        <nav>
          <a href="#features">Features</a>
          <a href="#workflow">Workflow</a>
          <Link to="/dashboard" className="nav-start">
            Login
          </Link>
        </nav>
      </header>

      {/* HERO */}
      <section className="hero-section">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="hero-left"
        >
          <span className="hero-badge">AI MATERIALS GENERATION</span>

          <h1>
            Generate Novel
            <br />
            Crystal Structures
            <br />
            Using AI
          </h1>

          <p>
            Scientific crystal generation powered by CDVAE, PyTorch,
            and advanced materials intelligence.
          </p>

          <div className="hero-buttons">
            <Link to="/dashboard" className="primary-btn">
              Launch Platform
            </Link>

            <Link to="/login" className="secondary-btn">
              Login
            </Link>
          </div>

          <div className="hero-stats">
            <div>
              <h3>10K+</h3>
              <span>Structures</span>
            </div>

            <div>
              <h3>3D</h3>
              <span>Visualization</span>
            </div>

            <div>
              <h3>AI</h3>
              <span>Generation</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7 }}
          className="hero-right"
        >
          <div className="viewer-preview">
            <div className="preview-header">
              <div>
                <h3>SiCSeO</h3>
                <p>Generated Crystal Structure</p>
              </div>

              <span className="status-pill">VALID</span>
            </div>

            <div className="fake-viewer">
              <div className="atom atom-1"></div>
              <div className="atom atom-2"></div>
              <div className="atom atom-3"></div>
              <div className="atom atom-4"></div>
              <div className="atom atom-5"></div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* FEATURES */}
      <section className="features-section" id="features">
        <div className="section-heading">
          <span>PLATFORM FEATURES</span>
          <h2>Scientific AI Infrastructure</h2>
        </div>

        <div className="feature-grid">
          <div className="feature-card">
            <Atom size={32} />
            <h3>AI Crystal Generation</h3>
            <p>
              Generate novel material structures using latent diffusion
              and CDVAE checkpoints.
            </p>
          </div>

          <div className="feature-card">
            <Boxes size={32} />
            <h3>3D Visualization</h3>
            <p>
              Interactive scientific visualization with lattice rendering
              and atomic controls.
            </p>
          </div>

          <div className="feature-card">
            <FlaskConical size={32} />
            <h3>Scientific Validation</h3>
            <p>
              Automated structural metrics, density analysis,
              and bond validation.
            </p>
          </div>

          <div className="feature-card">
            <Sparkles size={32} />
            <h3>CIF Export</h3>
            <p>
              Export production-ready CIF and JSON files for materials workflows.
            </p>
          </div>
        </div>
      </section>

      {/* WORKFLOW */}
      <section className="workflow-section" id="workflow">
        <div className="section-heading">
          <span>WORKFLOW</span>
          <h2>How It Works</h2>
        </div>

        <div className="workflow-grid">
          <div className="workflow-step">
            <div className="step-number">01</div>
            <h3>Select Constraints</h3>
            <p>Choose elements, lattice parameters, and scientific rules.</p>
          </div>

          <div className="workflow-step">
            <div className="step-number">02</div>
            <h3>Generate Structure</h3>
            <p>Run CDVAE inference using trained material checkpoints.</p>
          </div>

          <div className="workflow-step">
            <div className="step-number">03</div>
            <h3>Validate Crystal</h3>
            <p>Analyze structure validity and scientific metrics.</p>
          </div>

          <div className="workflow-step">
            <div className="step-number">04</div>
            <h3>Export Results</h3>
            <p>Download CIF and JSON outputs for research workflows.</p>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta-section">
        <h2>Start Generating Crystal Structures Today</h2>

        <p>
          Professional AI-driven materials discovery platform for scientific research.
        </p>

      </section>

      {/* FOOTER */}
      <footer className="landing-footer">
        <p>© 2026 CDVAE Lab. All rights reserved.</p>
      </footer>
    </div>
  );
}
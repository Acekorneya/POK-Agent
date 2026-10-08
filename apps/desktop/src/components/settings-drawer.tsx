import React from "react";

export type SettingsSection = "general" | "model" | "voice" | "system1" | "memory" | "tools" | "mcp";

export const SETTINGS_SECTIONS: { id: SettingsSection; label: string; hint: string }[] = [
  { id: "general", label: "General", hint: "Appearance, workspace, permissions" },
  { id: "model", label: "Model", hint: "Provider, API key, request tuning" },
  { id: "voice", label: "Voice", hint: "Microphone and speech to text" },
  { id: "system1", label: "System 1", hint: "Fast decision router" },
  { id: "mcp", label: "MCP servers", hint: "Connect external tool servers" },
  { id: "memory", label: "Memory & skills", hint: "What the agent has learned" },
  { id: "tools", label: "Generated tools", hint: "Helpers the agent built" },
];

/** Settings window: a section list on the left and one page per section. */
export function SettingsDrawer({ open, section, onSection, onClose, children }: {
  open: boolean; section: SettingsSection; onSection: (section: SettingsSection) => void;
  onClose: () => void; children: React.ReactNode;
}) {
  const current = SETTINGS_SECTIONS.find((item) => item.id === section) ?? SETTINGS_SECTIONS[0];
  return <>
    <button type="button" className="settings-backdrop" aria-label="Close settings" tabIndex={open ? 0 : -1} onClick={onClose} />
    <aside className="panel settings-panel" id="settings-panel" role="dialog" aria-modal={open} aria-labelledby="settings-heading">
      <nav className="settings-nav" aria-label="Settings sections">
        <h2 id="settings-heading">Settings</h2>
        {SETTINGS_SECTIONS.map((item) => <button key={item.id} type="button" className={item.id === section ? "active" : ""}
          aria-current={item.id === section ? "page" : undefined} onClick={() => onSection(item.id)}>
          <strong>{item.label}</strong><small>{item.hint}</small>
        </button>)}
      </nav>
      <div className="settings-page">
        <header className="settings-page-heading">
          <div><h3>{current.label}</h3><p>{current.hint}</p></div>
          <button type="button" className="settings-close" onClick={onClose} aria-label="Close settings">×</button>
        </header>
        <div className="settings-page-body">{children}</div>
      </div>
    </aside>
  </>;
}

/** A titled group of settings. */
export function SettingsCard({ title, description, status, actions, children }: {
  title: string; description?: React.ReactNode; status?: React.ReactNode; actions?: React.ReactNode; children?: React.ReactNode;
}) {
  return <section className="settings-card">
    <header>
      <div className="settings-card-title"><h4>{title}</h4>{status}</div>
      {actions && <div className="settings-card-actions">{actions}</div>}
    </header>
    {description && <p className="settings-card-description">{description}</p>}
    {children && <div className="settings-card-body">{children}</div>}
  </section>;
}

/** One setting: label and help on the left, the control on the right. */
export function SettingRow({ label, help, children }: { label: string; help?: React.ReactNode; children: React.ReactNode }) {
  return <div className="setting-row">
    <div className="setting-row-text"><span>{label}</span>{help && <small>{help}</small>}</div>
    <div className="setting-row-control">{children}</div>
  </div>;
}

export function StatusPill({ tone, children }: { tone: "ok" | "warn" | "off" | "error"; children: React.ReactNode }) {
  return <span className={`status-pill status-${tone}`}>{children}</span>;
}

/** A toggle switch backed by a real checkbox, so labels and tests work as usual. */
export function Switch({ label, checked, disabled, onChange }: {
  label: string; checked: boolean; disabled?: boolean; onChange: (checked: boolean) => void;
}) {
  return <label className="switch">
    <input type="checkbox" role="checkbox" aria-label={label} checked={checked} disabled={disabled}
      onChange={(event) => onChange(event.target.checked)} />
    <span className="switch-track" aria-hidden="true"><span className="switch-thumb" /></span>
  </label>;
}

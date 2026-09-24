import React from "react";
import { Appearance } from "./workspace";

export function SettingsDrawer({ open, onClose, children }: { open: boolean; onClose: () => void; children: React.ReactNode }) {
  return <>
    <button type="button" className="settings-backdrop" aria-label="Close settings" tabIndex={open ? 0 : -1} onClick={onClose} />
    <aside className="panel settings-panel" id="settings-panel" aria-label="Agent settings">
      <div className="settings-panel-heading">
        <h2 id="settings-heading">Settings</h2>
        <button type="button" className="settings-close" onClick={onClose} aria-label="Close settings">×</button>
      </div>
      <Appearance />
      {children}
    </aside>
  </>;
}

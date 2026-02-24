"use client";
import { Sun, Moon, Trash2, PanelLeft } from "lucide-react";
import styles from "./TopBar.module.css";

interface Props {
  theme: "dark" | "light";
  onToggleTheme: () => void;
  onToggleSidebar: () => void;
  onClear: () => void;
  hasMessages: boolean;
  apiOnline: boolean;
}

export default function TopBar({
  theme,
  onToggleTheme,
  onToggleSidebar,
  onClear,
  hasMessages,
  apiOnline,
}: Props) {
  return (
    <div className={styles.topbar}>
      <div className={styles.left}>
        <button
          className={styles.iconBtn}
          onClick={onToggleSidebar}
          title="Toggle sidebar"
        >
          <PanelLeft size={17} />
        </button>
        <div className={styles.title}>AI Assistant</div>
        <div className={styles.modelBadge}>qwen3-vl</div>
      </div>

      <div className={styles.right}>
        {!apiOnline && (
          <span className={styles.offlineWarning}>API Offline</span>
        )}
        {hasMessages && (
          <button
            className={styles.iconBtn}
            onClick={onClear}
            title="Clear chat"
          >
            <Trash2 size={15} />
            <span>Clear</span>
          </button>
        )}
        <button
          className={styles.themeBtn}
          onClick={onToggleTheme}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </div>
  );
}

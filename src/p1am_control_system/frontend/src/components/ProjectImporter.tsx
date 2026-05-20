import React, { useState } from "react";
import { Upload, FileArchive, CheckCircle, Loader2 } from "lucide-react";

interface ProjectImporterProps {
  onImportSuccess: (summary: ImportSummary) => void;
  triggerNotification: (msg: string, type: "success" | "error" | "info") => void;
}

export interface ImportSummary {
  status: string;
  tags_imported: number;
  mapped_registers: number;
  areas_created: string[];
  units_created: string[];
  equipment_created: string[];
}

export const ProjectImporter: React.FC<ProjectImporterProps> = ({
  onImportSuccess,
  triggerNotification,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [dragOver, setDragOver] = useState<boolean>(false);
  const [summary, setSummary] = useState<ImportSummary | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith(".zip")) {
        setFile(droppedFile);
        setSummary(null);
      } else {
        triggerNotification("Only ZIP files are supported.", "error");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith(".zip")) {
        setFile(selectedFile);
        setSummary(null);
      } else {
        triggerNotification("Only ZIP files are supported.", "error");
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/project/import", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data: ImportSummary = await res.json();
        setSummary(data);
        onImportSuccess(data);
        triggerNotification("Project imported successfully!", "success");
      } else {
        const errorData = await res.json();
        triggerNotification(
          `Import failed: ${errorData.detail || "Unknown error"}`,
          "error"
        );
      }
    } catch (err) {
      triggerNotification("Connection error importing project configuration.", "error");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: "1.5rem" }}>
      <div className="panel-header" style={{ marginBottom: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <FileArchive size={18} color="var(--accent-cyan)" />
          <span style={{ fontWeight: 700, fontSize: "0.95rem" }}>SCADA Project Config Importer</span>
        </div>
      </div>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${dragOver ? "var(--accent-cyan)" : "var(--panel-border)"}`,
          borderRadius: "8px",
          padding: "2rem 1rem",
          textAlign: "center",
          backgroundColor: dragOver ? "rgba(56, 189, 248, 0.05)" : "rgba(255, 255, 255, 0.01)",
          transition: "all var(--transition-fast)",
          cursor: "pointer",
          position: "relative",
        }}
      >
        <input
          type="file"
          accept=".zip"
          onChange={handleFileChange}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            opacity: 0,
            cursor: "pointer",
          }}
        />
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.75rem" }}>
          <Upload size={32} color={dragOver ? "var(--accent-cyan)" : "var(--text-secondary)"} />
          <div>
            <p style={{ fontWeight: 600, fontSize: "0.85rem", color: "var(--text-primary)" }}>
              Drag & drop your PLC config ZIP here, or click to browse
            </p>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
              Accepts SCADA archives containing tagl.json and driver .SDV files
            </p>
          </div>
          {file && (
            <div
              style={{
                marginTop: "0.5rem",
                padding: "0.4rem 0.8rem",
                background: "rgba(255, 255, 255, 0.03)",
                border: "1px solid var(--panel-border)",
                borderRadius: "4px",
                fontSize: "0.75rem",
                fontWeight: 600,
                color: "var(--accent-cyan)",
              }}
            >
              Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)
            </div>
          )}
        </div>
      </div>

      {file && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "1rem" }}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleUpload}
            disabled={uploading}
            style={{
              padding: "0.5rem 1.25rem",
              fontSize: "0.8rem",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
            }}
          >
            {uploading ? (
              <>
                <Loader2 className="animate-spin" size={14} />
                Importing...
              </>
            ) : (
              "Ingest Config & Re-initialize"
            )}
          </button>
        </div>
      )}

      {summary && (
        <div
          style={{
            marginTop: "1.25rem",
            padding: "1rem",
            background: "rgba(16, 185, 129, 0.03)",
            border: "1px solid rgba(16, 185, 129, 0.2)",
            borderRadius: "6px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <CheckCircle size={16} color="var(--color-success)" />
            <span style={{ fontWeight: 700, fontSize: "0.85rem", color: "var(--color-success)" }}>
              Import Complete
            </span>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "0.75rem",
              fontSize: "0.8rem",
              color: "var(--text-secondary)",
            }}
          >
            <div>
              <strong>Tags Imported:</strong> {summary.tags_imported}
            </div>
            <div>
              <strong>Mapped Registers:</strong> {summary.mapped_registers}
            </div>
            <div style={{ gridColumn: "span 2" }}>
              <strong>Areas Created:</strong>{" "}
              {summary.areas_created.length > 0 ? summary.areas_created.join(", ") : "None"}
            </div>
            <div style={{ gridColumn: "span 2" }}>
              <strong>Units Created:</strong>{" "}
              {summary.units_created.length > 0 ? summary.units_created.join(", ") : "None"}
            </div>
            <div style={{ gridColumn: "span 2" }}>
              <strong>Equipment Created:</strong>{" "}
              {summary.equipment_created.length > 0 ? summary.equipment_created.join(", ") : "None"}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

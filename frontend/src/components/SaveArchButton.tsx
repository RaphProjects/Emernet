import { useState } from 'react';

interface SaveProps {
  archId: string;
  defaultName?: string;
}


export default function SaveArchButton({ archId, defaultName = "my_arch" }: SaveProps) {
  const [filename, setFilename] = useState(defaultName);
  const [status, setStatus] = useState<"idle" | "saving" | "success" | "error">("idle");
  const [message, setMessage] = useState("");

  const handleSave = async () => {
    if (!archId) return;
    setStatus("saving");
    
    try {
      {/* Send POST request with query parameters */}
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/api/save_arch?arch_id=${archId}&filename=${filename}`, 
        {
          method: 'POST'
        }
      );
      const data = await response.json();
      
      if (data.error) {
        setStatus("error");
        setMessage(data.error);
      } else {
        setStatus("success");
        setMessage(data.message);
        {/* Reset after 3 seconds */}
        setTimeout(() => setStatus("idle"), 3000); 
      }
    } catch (err) {
      setStatus("error");
      setMessage("Network error");
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <input 
        type="text" 
        value={filename} 
        onChange={(e) => setFilename(e.target.value)}
        style={{ padding: '6px 10px', borderRadius: '4px', border: '1px solid #475569', background: '#0f172a', color: 'white', width: '150px', fontSize: '13px' }}
        placeholder="filename"
      />
      <button 
        className="btn" 
        style={{ background: '#10b981', color: 'white' }} 
        onClick={handleSave}
        disabled={status === "saving"}
      >
        {status === "saving" ? "Saving..." : "Save"}
      </button>
      
      {status === "success" && <span style={{ color: '#10b981', fontSize: '12px' }}>Done: {message}</span>}
      {status === "error" && <span style={{ color: '#ef4444', fontSize: '12px' }}>Error: {message}</span>}
    </div>
  );
}
import React from "react";
import { ShieldAlert, RefreshCw } from "lucide-react";

interface EStopButtonProps {
  eStopActive: boolean;
  onTriggerEStop: () => void;
  onClearEStop: () => void;
}

export const EStopButton: React.FC<EStopButtonProps> = ({
  eStopActive,
  onTriggerEStop,
  onClearEStop,
}) => {
  return (
    <div className="flex flex-col items-center gap-2">
      {eStopActive ? (
        <button
          onClick={onClearEStop}
          className="flex items-center gap-2 px-6 py-3 bg-yellow-600 hover:bg-yellow-500 text-white rounded-xl font-bold transition-all shadow-[0_0_20px_rgba(202,138,4,0.4)] animate-pulse"
        >
          <RefreshCw className="w-5 h-5" />
          CLEAR E-STOP
        </button>
      ) : (
        <button
          onClick={onTriggerEStop}
          className="flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-500 text-white rounded-xl font-bold transition-all shadow-[0_0_20px_rgba(220,38,38,0.4)] hover:shadow-[0_0_30px_rgba(220,38,38,0.6)] hover:scale-105 active:scale-95"
        >
          <ShieldAlert className="w-5 h-5" />
          EMERGENCY STOP
        </button>
      )}
      {eStopActive && (
        <span className="text-red-400 font-semibold text-sm">
          System is safety-locked.
        </span>
      )}
    </div>
  );
};

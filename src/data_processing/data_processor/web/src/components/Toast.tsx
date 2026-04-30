import { memo, useEffect, useState } from 'react';
import {
  AlertCircle,
  CheckCircle,
  AlertTriangle,
  Info,
  X,
} from 'lucide-react';
import type { NotificationType, Notification } from '../hooks/useNotification';

interface ToastProps {
  notification: Notification;
  onClose: (id: string) => void;
}

const Toast = memo(function Toast({ notification, onClose }: ToastProps) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (!notification.duration || notification.duration <= 0) {
      return;
    }

    const timer = setTimeout(() => {
      setIsVisible(false);
    }, notification.duration);

    return () => clearTimeout(timer);
  }, [notification.duration]);

  const getIcon = (type: NotificationType) => {
    const iconProps = 'w-5 h-5';
    switch (type) {
      case 'error':
        return <AlertCircle className={`${iconProps} text-red-400`} />;
      case 'success':
        return <CheckCircle className={`${iconProps} text-green-400`} />;
      case 'warning':
        return <AlertTriangle className={`${iconProps} text-yellow-400`} />;
      case 'info':
      default:
        return <Info className={`${iconProps} text-blue-400`} />;
    }
  };

  const getBackgroundColor = (type: NotificationType) => {
    switch (type) {
      case 'error':
        return 'bg-red-900/20 border-red-500/50';
      case 'success':
        return 'bg-green-900/20 border-green-500/50';
      case 'warning':
        return 'bg-yellow-900/20 border-yellow-500/50';
      case 'info':
      default:
        return 'bg-blue-900/20 border-blue-500/50';
    }
  };

  const getTextColor = (type: NotificationType) => {
    switch (type) {
      case 'error':
        return 'text-red-300';
      case 'success':
        return 'text-green-300';
      case 'warning':
        return 'text-yellow-300';
      case 'info':
      default:
        return 'text-blue-300';
    }
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div
      className={`
        flex items-start gap-3 p-4 rounded-lg border
        ${getBackgroundColor(notification.type)}
        ${getTextColor(notification.type)}
        backdrop-blur-sm
        animate-slide-in
        shadow-lg
      `}
      role="alert"
      aria-live="polite"
      aria-atomic="true"
    >
      {getIcon(notification.type)}
      <div className="flex-1 text-sm font-medium">
        {notification.message}
      </div>
      <button
        onClick={() => {
          setIsVisible(false);
          onClose(notification.id);
        }}
        className="p-1 hover:bg-white/10 rounded transition-colors flex-shrink-0"
        aria-label="Close notification"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
});

interface ToastContainerProps {
  notifications: Notification[];
  onClose: (id: string) => void;
}

export const ToastContainer = memo(function ToastContainer({
  notifications,
  onClose,
}: ToastContainerProps) {
  return (
    <div
      className="fixed bottom-6 right-6 space-y-3 z-50 pointer-events-auto"
      role="region"
      aria-label="Notifications"
    >
      {notifications.map((notification) => (
        <Toast
          key={notification.id}
          notification={notification}
          onClose={onClose}
        />
      ))}
    </div>
  );
});

export default ToastContainer;

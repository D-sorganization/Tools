import { useCallback, useState } from 'react';

export type NotificationType = 'error' | 'success' | 'warning' | 'info';

export interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  duration?: number;
}

interface UseNotificationReturn {
  notifications: Notification[];
  showNotification: (options: {
    type: NotificationType;
    message: string;
    duration?: number;
  }) => void;
  removeNotification: (id: string) => void;
  clearAll: () => void;
}

/**
 * Hook for managing toast notifications.
 *
 * Usage:
 *   const { showNotification } = useNotification();
 *   showNotification({
 *     type: 'error',
 *     message: 'File too large',
 *     duration: 5000
 *   });
 */
export function useNotification(): UseNotificationReturn {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const removeNotification = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const showNotification = useCallback(
    (options: {
      type: NotificationType;
      message: string;
      duration?: number;
    }) => {
      const id = `notification-${Date.now()}-${Math.random()}`;
      const duration = options.duration ?? 5000; // Default 5 seconds

      setNotifications((prev) => [...prev, { id, ...options }]);

      // Auto-remove notification after duration
      if (duration > 0) {
        const timer = setTimeout(() => {
          removeNotification(id);
        }, duration);

        // Store timer reference for potential cleanup
        return () => clearTimeout(timer);
      }
    },
    [removeNotification]
  );

  const clearAll = useCallback(() => {
    setNotifications([]);
  }, []);

  return {
    notifications,
    showNotification,
    removeNotification,
    clearAll,
  };
}

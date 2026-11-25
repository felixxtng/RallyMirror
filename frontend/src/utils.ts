export const formatTime = (seconds: number) => {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
};
export const formatBytes = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
export const finiteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

export async function videoMetadata(
  file: File,
): Promise<{ duration: number; width: number; height: number }> {
  if (!file.name.toLowerCase().endsWith('.mp4'))
    throw new Error('Choose an MP4 video. Other formats are not supported yet.');
  if (file.size === 0) throw new Error('This video is empty. Choose another file.');
  if (file.size > 200 * 1024 * 1024) throw new Error('Choose a video smaller than 200 MiB.');
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const url = URL.createObjectURL(file);
    const cleanup = () => {
      clearTimeout(timer);
      video.onloadedmetadata = null;
      video.onerror = null;
      video.removeAttribute('src');
      video.load();
      URL.revokeObjectURL(url);
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('Could not read this video. Try another MP4 file.'));
    }, 10000);
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      const result = {
        duration: video.duration,
        width: video.videoWidth,
        height: video.videoHeight,
      };
      cleanup();
      if (!Number.isFinite(result.duration) || result.duration <= 0)
        reject(new Error('This video has an invalid duration.'));
      else if (result.duration > 60)
        reject(new Error('Choose a rally clip up to 60 seconds long.'));
      else resolve(result);
    };
    video.onerror = () => {
      cleanup();
      reject(
        new Error('Your browser could not read this MP4. Re-export it as H.264 MP4 and try again.'),
      );
    };
    video.src = url;
  });
}

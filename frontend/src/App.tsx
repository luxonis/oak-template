import { Streams } from '@luxonis/depthai-viewer-common';

const DEFAULT_TOPICS = ['Video', 'Visualizations'];

export function App() {
  return (
    <main>
      <h1>OAK Template</h1>
      <Streams defaultTopics={DEFAULT_TOPICS} numberOfColumns={2} />
    </main>
  );
}

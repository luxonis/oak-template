import '@luxonis/depthai-pipeline-lib/styles';
import '@luxonis/depthai-viewer-common/styles';
import { DepthAIContext } from '@luxonis/depthai-viewer-common';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DepthAIContext>
      <App />
    </DepthAIContext>
  </StrictMode>,
);

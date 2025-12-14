import React from 'react';
import ImageUpload from './ImageUpload';
import ReportViewer from './ReportViewer';
import { useStore } from 'zustand/store';  // Assuming Zustand setup

const Dashboard = () => {
  const { patients } = useStore();

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold">MediScan AI Dashboard</h1>
      <ImageUpload />
      <ReportViewer />
      {/* Analytics with Recharts */}
      <div>
        {/* 3D viz with Three.js for CT/MRI */}
      </div>
    </div>
  );
};

export default Dashboard;

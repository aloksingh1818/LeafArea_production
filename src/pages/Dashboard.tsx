import React, { useState } from 'react';
import PlantDiseasePredictor from './PlantDiseasePredictor';
import LeafAreaDetails from './LeafAreaDetails';

const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'predict' | 'leafarea' | 'batch' | 'model' | 'about'>('predict');

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <aside className="w-64 bg-green-800 text-white flex flex-col py-6 px-4">
        <div className="text-2xl font-bold mb-8 text-center tracking-wide">🌱 LeafAI</div>
        <nav className="flex flex-col gap-4">
          <button className={`text-left px-3 py-2 rounded transition ${activeTab==='predict' ? 'bg-green-600' : 'hover:bg-green-700'}`} onClick={()=>setActiveTab('predict')}>Predict Disease</button>
          <button className={`text-left px-3 py-2 rounded transition ${activeTab==='leafarea' ? 'bg-green-600' : 'hover:bg-green-700'}`} onClick={()=>setActiveTab('leafarea')}>Leaf Area Details</button>
          <button className={`text-left px-3 py-2 rounded transition ${activeTab==='batch' ? 'bg-green-600' : 'hover:bg-green-700'}`} onClick={()=>setActiveTab('batch')}>Batch Prediction</button>
          <button className={`text-left px-3 py-2 rounded transition ${activeTab==='model' ? 'bg-green-600' : 'hover:bg-green-700'}`} onClick={()=>setActiveTab('model')}>Model Info</button>
          <button className={`text-left px-3 py-2 rounded transition ${activeTab==='about' ? 'bg-green-600' : 'hover:bg-green-700'}`} onClick={()=>setActiveTab('about')}>About</button>
        </nav>
        <div className="mt-auto text-xs text-center text-green-200 pt-8">&copy; {new Date().getFullYear()} LeafAI</div>
      </aside>
      {/* Main Content */}
      <main className="flex-1 p-8">
        <header className="mb-8 flex items-center justify-between">
          <h1 className="text-3xl font-bold text-green-900">Plant Disease Detection Dashboard</h1>
          <div className="bg-green-100 text-green-800 px-4 py-2 rounded shadow text-sm font-semibold">Model Accuracy: 69%</div>
        </header>
        <section>
          {activeTab === 'predict' && <PlantDiseasePredictor />}
          {activeTab === 'leafarea' && <LeafAreaDetails />}
          {activeTab === 'batch' && (
            <div className="bg-white rounded shadow p-8 text-center text-gray-600">
              <h2 className="text-xl font-bold mb-4 text-green-700">Batch Prediction</h2>
              <p>Upload a folder of images to predict diseases in bulk. (Coming soon!)</p>
            </div>
          )}
          {activeTab === 'model' && (
            <div className="bg-white rounded shadow p-8">
              <h2 className="text-xl font-bold mb-4 text-green-700">Model Information</h2>
              <ul className="list-disc ml-6 text-gray-700">
                <li>Architecture: Custom CNN (Keras)</li>
                <li>Input Size: 128x128 RGB</li>
                <li>Classes: 15 (Pepper, Potato, Tomato diseases & healthy)</li>
                <li>Accuracy: 69%</li>
                <li>Framework: FastAPI + React</li>
              </ul>
            </div>
          )}
          {activeTab === 'about' && (
            <div className="bg-white rounded shadow p-8">
              <h2 className="text-xl font-bold mb-4 text-green-700">About & Developers</h2>
              <p className="mb-2">This project uses deep learning to detect plant diseases from leaf images. Built with FastAPI, TensorFlow, and React.</p>
              <p>Developed by your team. For more info, see the README or contact us.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default Dashboard;

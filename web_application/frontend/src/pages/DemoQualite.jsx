import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowBigLeft, Upload, CheckCircle, AlertTriangle, ShieldCheck, Camera, Activity } from 'lucide-react';

const DemoQualite = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState({ type: '', message: '' });

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleEvaluate = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setStatus({ type: 'info', message: 'Analyse biométrique en cours...' });

    const formData = new FormData();
    // ⚠️ IMPORTANT : Ton backend attend la clé 'photo'
    formData.append('photo', file);

    try {
      // ⚠️ IMPORTANT : Appel de la nouvelle route
      const response = await fetch('http://localhost:8000/api/quality', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setResult(data);
        // Ton backend renvoie "bonne", "acceptable", ou "a_remplacer"
        const isValid = data.recommendation !== "a_remplacer";
        setStatus({ 
          type: isValid ? 'success' : 'error', 
          message: isValid ? 'Image validée par le système.' : 'Image rejetée par le filtre de qualité.' 
        });
      } else {
        setStatus({ type: 'error', message: data.detail || "Erreur lors de l'analyse." });
      }
    } catch (error) {
      setStatus({ type: 'error', message: "Impossible de contacter le serveur backend." });
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-24 px-6 md:px-20 pb-12">
      <div className="max-w-4xl mx-auto bg-white rounded-3xl shadow-xl overflow-hidden">
        
        <div className="bg-blue-700 p-8 text-white">
          <Link to="/" className="inline-flex items-center text-blue-200 hover:text-white mb-4 transition-colors">
            <ArrowBigLeft className="w-5 h-5 mr-2" /> Retour à l'accueil
          </Link>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <ShieldCheck className="w-8 h-8" />
            Audit de Qualité (FIQA)
          </h1>
          <p className="mt-2 text-blue-100">
            Testez la robustesse de notre pipeline. Uploadez une image floue, sombre ou mal cadrée pour voir comment l'IA réagit.
          </p>
        </div>

        <div className="p-8 grid md:grid-cols-2 gap-12">
          
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
              <Camera className="text-blue-600 w-6 h-6" />
              Soumettre une image
            </h2>
            
            <form onSubmit={handleEvaluate} className="space-y-4">
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center hover:bg-gray-50 transition-colors">
                <input 
                  type="file" accept="image/*"
                  onChange={handleFileChange}
                  className="hidden" id="quality-upload"
                />
                <label htmlFor="quality-upload" className="cursor-pointer flex flex-col items-center">
                  {previewUrl ? (
                    <img src={previewUrl} alt="Preview" className="h-48 object-contain rounded-lg shadow-sm mb-4" />
                  ) : (
                    <Upload className="w-12 h-12 text-blue-500 mb-4" />
                  )}
                  <span className="text-gray-700 font-medium">
                    {file ? 'Changer d\'image' : 'Sélectionner une image'}
                  </span>
                </label>
              </div>
              <button 
                disabled={loading || !file} 
                className={`w-full py-3 rounded-xl font-medium transition-colors ${loading || !file ? 'bg-gray-300 text-gray-500 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 text-white'}`}
              >
                {loading ? 'Analyse en cours...' : 'Lancer le diagnostic'}
              </button>
            </form>

            {status.message && (
              <div className={`p-4 rounded-lg flex items-center gap-3 ${
                status.type === 'error' ? 'bg-red-100 text-red-800 border border-red-200' : 
                status.type === 'success' ? 'bg-green-100 text-green-800 border border-green-200' : 
                'bg-blue-100 text-blue-800 border border-blue-200'
              }`}>
                {status.type === 'error' ? <AlertTriangle className="w-5 h-5"/> : <CheckCircle className="w-5 h-5"/>}
                <span>{status.message}</span>
              </div>
            )}
          </div>

          <div className="space-y-6 bg-gray-50 p-6 rounded-2xl border border-gray-100">
            <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
              <Activity className="text-blue-600 w-6 h-6" />
              Résultats de l'analyse
            </h2>
            
            {result ? (
              <div className="space-y-6">
                <div className="text-center">
                  <div className={`inline-flex items-center justify-center w-32 h-32 rounded-full border-8 mb-4 ${result.recommendation !== 'a_remplacer' ? 'border-green-500 text-green-600' : 'border-red-500 text-red-600'}`}>
                    <span className="text-3xl font-bold">{result.final_score}<span className="text-lg">/100</span></span>
                  </div>
                  <h3 className={`text-xl font-bold uppercase ${result.recommendation !== 'a_remplacer' ? 'text-green-600' : 'text-red-600'}`}>
                    {result.recommendation.replace('_', ' ')}
                  </h3>
                </div>

                <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200 text-sm text-gray-700 space-y-2">
                  <p className="font-semibold border-b pb-2 mb-2">Détails du moteur hybride :</p>
                  <p><b>Netteté (OpenCV) :</b> {result.sharpness_score}/100</p>
                  <p><b>Lumière (OpenCV) :</b> {result.brightness_score}/100</p>
                  <p><b>Géométrie (GraFIQs) :</b> {result.grafiqs_score}/100</p>
                  <p><b>Visage détecté :</b> {result.face_detected ? 'Oui' : 'Non (Mode Fallback)'}</p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-48 text-gray-400 text-center">
                <ShieldCheck className="w-12 h-12 mb-2 opacity-50" />
                <p>En attente d'une image pour afficher le diagnostic.</p>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
};

export default DemoQualite;
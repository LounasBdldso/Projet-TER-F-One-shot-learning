import React, { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import { ArrowBigLeft, Upload, Image as ImageIcon, CheckCircle, AlertTriangle, Scan } from 'lucide-react';

// ============================================================
// Composant ScanEffect — laser qui scanne la photo de groupe
// ============================================================
const ScanEffect = ({ imageUrl }) => (
  <div className="relative rounded-xl overflow-hidden shadow-lg border-2 border-blue-400">
    {/* Image de fond */}
    <img src={imageUrl} alt="Scan en cours" className="w-full h-auto opacity-80" />

    {/* Overlay sombre */}
    <div className="absolute inset-0 bg-blue-950/30" />

    {/* Ligne laser qui descend */}
    <div
      className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-blue-400 to-transparent shadow-[0_0_18px_4px_rgba(96,165,250,0.8)]"
      style={{
        animation: 'scanLine 2s ease-in-out infinite',
        top: 0,
      }}
    />

    {/* Halo au-dessus du laser */}
    <div
      className="absolute left-0 right-0 h-16 bg-gradient-to-b from-blue-400/20 to-transparent"
      style={{
        animation: 'scanLine 2s ease-in-out infinite',
        top: 0,
      }}
    />

    {/* Coins animés */}
    <div className="absolute top-2 left-2 w-6 h-6 border-t-2 border-l-2 border-blue-400 rounded-tl" />
    <div className="absolute top-2 right-2 w-6 h-6 border-t-2 border-r-2 border-blue-400 rounded-tr" />
    <div className="absolute bottom-2 left-2 w-6 h-6 border-b-2 border-l-2 border-blue-400 rounded-bl" />
    <div className="absolute bottom-2 right-2 w-6 h-6 border-b-2 border-r-2 border-blue-400 rounded-br" />

    {/* Label */}
    <div className="absolute bottom-4 left-0 right-0 flex justify-center">
      <span className="bg-blue-900/80 text-blue-200 text-xs px-3 py-1 rounded-full
        flex items-center gap-2 backdrop-blur-sm border border-blue-400/30">
        <Scan className="w-3 h-3 animate-pulse" />
        Analyse biométrique en cours...
      </span>
    </div>

    {/* Animation CSS */}
    <style>{`
      @keyframes scanLine {
        0%   { transform: translateY(0); opacity: 1; }
        45%  { opacity: 1; }
        50%  { transform: translateY(calc(var(--img-h, 400px) - 4px)); opacity: 0.8; }
        51%  { opacity: 0; transform: translateY(0); }
        55%  { opacity: 1; }
        100% { transform: translateY(0); opacity: 1; }
      }
    `}</style>
  </div>
);


// ============================================================
// Page principale
// ============================================================
const DemoReconnaissance = () => {
  const [referenceFiles, setReferenceFiles]   = useState([]);
  const [groupFile, setGroupFile]             = useState(null);
  const [groupPreview, setGroupPreview]       = useState(null);
  const [status, setStatus]                   = useState({ type: '', message: '' });
  const [resultImage, setResultImage]         = useState(null);
  const [loading, setLoading]                 = useState(false);

  const handleGroupFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setGroupFile(file);
      setGroupPreview(URL.createObjectURL(file));
      setResultImage(null);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (referenceFiles.length === 0) {
      setStatus({ type: 'error', message: 'Veuillez ajouter au moins une photo de référence.' });
      return;
    }
    if (!groupFile) {
      setStatus({ type: 'error', message: 'Veuillez ajouter une photo de groupe.' });
      return;
    }

    setLoading(true);
    setResultImage(null);
    setStatus({ type: 'info', message: 'Analyse biométrique en cours...' });

    const formData = new FormData();
    formData.append('group_photo', groupFile);

    for (let i = 0; i < referenceFiles.length; i++) {
      const file = referenceFiles[i];
      formData.append('reference_photos', file);
      const name = file.name.substring(0, file.name.lastIndexOf('.')) || file.name;
      formData.append('reference_names', name);
    }

    try {
      const response = await fetch('http://localhost:8000/api/recognize', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob         = await response.blob();
        const imageUrl     = URL.createObjectURL(blob);
        const facesDetected = response.headers.get('X-Faces-Detected');

        setResultImage(imageUrl);
        setStatus({
          type: 'success',
          message: `Reconnaissance terminée ! ${facesDetected} visage(s) détecté(s).`,
        });
      } else {
        const data = await response.json();
        setStatus({ type: 'error', message: data.detail || "Erreur lors de la prédiction." });
      }
    } catch (error) {
      setStatus({ type: 'error', message: "Impossible de contacter le serveur backend." });
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-24 px-6 md:px-20 pb-12">
      <div className="max-w-5xl mx-auto bg-white rounded-3xl shadow-xl overflow-hidden">

        {/* Header */}
        <div className="bg-blue-900 p-8 text-white">
          <Link
            to="/"
            className="inline-flex items-center text-blue-200 hover:text-white mb-4 transition-colors"
          >
            <ArrowBigLeft className="w-5 h-5 mr-2" /> Retour à l'accueil
          </Link>
          <h1 className="text-3xl font-bold">Démonstration : Reconnaissance</h1>
          <p className="mt-2 text-blue-100">
            Architecture One-Shot : uploadez vos références et la photo de groupe pour une
            identification instantanée.
          </p>
        </div>

        {/* Bandeau de statut */}
        {status.message && (
          <div
            className={`p-4 mx-8 mt-8 rounded-lg flex items-center gap-3 ${
              status.type === 'error'
                ? 'bg-red-100 text-red-800 border border-red-200'
                : status.type === 'success'
                ? 'bg-green-100 text-green-800 border border-green-200'
                : 'bg-blue-100 text-blue-800 border border-blue-200'
            }`}
          >
            {status.type === 'error' ? (
              <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            ) : (
              <CheckCircle className="w-5 h-5 flex-shrink-0" />
            )}
            <span>{status.message}</span>
          </div>
        )}

        <form onSubmit={handlePredict} className="p-8 space-y-12">
          <div className="grid md:grid-cols-2 gap-12">

            {/* Étape 1 — Références */}
            <div className="space-y-4">
              <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                  Étape 1
                </span>
                Références (Support Set)
              </h2>
              <p className="text-sm text-gray-500">
                Nommez les fichiers avec le prénom de la personne (ex :{' '}
                <b>Lounas.jpg</b>).
              </p>

              <div className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center
                hover:bg-gray-50 transition-colors">
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={(e) => setReferenceFiles(Array.from(e.target.files))}
                  className="hidden"
                  id="support-upload"
                />
                <label htmlFor="support-upload" className="cursor-pointer flex flex-col items-center">
                  <Upload className="w-8 h-8 text-blue-500 mb-2" />
                  <span className="text-gray-700 font-medium">
                    Sélectionner des références
                  </span>
                  <span className="text-xs text-gray-400 mt-1">
                    {referenceFiles.length > 0
                      ? `${referenceFiles.length} fichier(s) prêt(s)`
                      : 'Aucun fichier'}
                  </span>
                </label>
              </div>

              {/* Miniatures des références */}
              {referenceFiles.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                  {referenceFiles.map((f, i) => (
                    <div key={i} className="relative group">
                      <img
                        src={URL.createObjectURL(f)}
                        alt={f.name}
                        className="w-14 h-14 object-cover rounded-lg border-2 border-blue-200 shadow-sm"
                      />
                      <span className="absolute -bottom-5 left-0 right-0 text-center text-[10px]
                        text-gray-500 truncate w-14">
                        {f.name.split('.')[0]}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Étape 2 — Photo de groupe */}
            <div className="space-y-4">
              <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                  Étape 2
                </span>
                Photo de Groupe
              </h2>
              <p className="text-sm text-gray-500">
                L'IA détectera les visages et les comparera aux références fournies.
              </p>

              <div className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center
                hover:bg-gray-50 transition-colors">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleGroupFileChange}
                  className="hidden"
                  id="group-upload"
                />
                <label htmlFor="group-upload" className="cursor-pointer flex flex-col items-center">
                  <ImageIcon className="w-8 h-8 text-blue-500 mb-2" />
                  <span className="text-gray-700 font-medium">
                    Sélectionner la photo de groupe
                  </span>
                  <span className="text-xs text-gray-400 mt-1">
                    {groupFile ? groupFile.name : 'Aucun fichier'}
                  </span>
                </label>
              </div>

              {/* Preview de la photo de groupe — avec scan si loading */}
              {groupPreview && !loading && !resultImage && (
                <div className="mt-2 rounded-xl overflow-hidden shadow-md border border-gray-200">
                  <img
                    src={groupPreview}
                    alt="Aperçu photo de groupe"
                    className="w-full h-auto"
                  />
                </div>
              )}

              {/* Effet SCAN pendant le chargement */}
              {groupPreview && loading && (
                <div className="mt-2">
                  <ScanEffect imageUrl={groupPreview} />
                </div>
              )}
            </div>
          </div>

          {/* Bouton lancer */}
          <button
            disabled={loading}
            className={`w-full py-4 rounded-xl font-bold text-lg shadow-lg transition-all
              flex items-center justify-center gap-3 ${
                loading
                  ? 'bg-blue-800 text-blue-300 cursor-not-allowed'
                  : 'bg-blue-900 hover:bg-blue-950 text-white'
              }`}
          >
            {loading ? (
              <>
                <Scan className="w-5 h-5 animate-pulse" />
                Analyse biométrique en cours...
              </>
            ) : (
              'Lancer la Reconnaissance'
            )}
          </button>
        </form>

        {/* Résultat */}
        {resultImage && (
          <div className="p-8 border-t border-gray-100 bg-gray-50">
            <h3 className="text-xl font-bold text-gray-800 mb-6 text-center">
              Résultat de la Reconnaissance
            </h3>
            <div className="rounded-xl overflow-hidden shadow-lg border-4 border-white
              mx-auto max-w-3xl">
              <img src={resultImage} alt="Résultat" className="w-full h-auto" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DemoReconnaissance;
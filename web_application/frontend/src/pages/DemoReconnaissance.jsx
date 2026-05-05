import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowBigLeft, Upload, Image as ImageIcon, CheckCircle, AlertTriangle } from 'lucide-react';

const DemoReconnaissance = () => {
  const [referenceFiles, setReferenceFiles] = useState([]);
  const [groupFile, setGroupFile] = useState(null);
  const [status, setStatus] = useState({ type: '', message: '' });
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);

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
    setStatus({ type: 'info', message: 'Analyse et comparaison en cours...' });

    // Construction du payload attendu par le nouveau backend FastAPI
    const formData = new FormData();
    formData.append('group_photo', groupFile);

    // On boucle sur les fichiers de référence pour les ajouter un par un
    for (let i = 0; i < referenceFiles.length; i++) {
      const file = referenceFiles[i];
      formData.append('reference_photos', file);
      // On utilise le nom du fichier (sans l'extension) comme "nom" de la personne
      const name = file.name.substring(0, file.name.lastIndexOf('.')) || file.name;
      formData.append('reference_names', name);
    }

    try {
      // Appel à la nouvelle route qui fait tout en une seule fois
      const response = await fetch('http://localhost:8000/api/recognize', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        // Le backend renvoie directement l'image JPEG annotée
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        
        // On peut même lire les headers personnalisés que tu as créés !
        const facesDetected = response.headers.get('X-Faces-Detected');
        
        setResultImage(imageUrl);
        setStatus({ type: 'success', message: `Reconnaissance terminée ! ${facesDetected} visage(s) détecté(s).` });
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
        
        <div className="bg-blue-900 p-8 text-white">
          <Link to="/" className="inline-flex items-center text-blue-200 hover:text-white mb-4 transition-colors">
            <ArrowBigLeft className="w-5 h-5 mr-2" /> Retour à l'accueil
          </Link>
          <h1 className="text-3xl font-bold">Démonstration : Reconnaissance</h1>
          <p className="mt-2 text-blue-100">Architecture Stateless : Uploadez vos références et la photo de groupe pour une analyse "One-Shot".</p>
        </div>

        {status.message && (
          <div className={`p-4 mx-8 mt-8 rounded-lg flex items-center gap-3 ${
            status.type === 'error' ? 'bg-red-100 text-red-800 border border-red-200' : 
            status.type === 'success' ? 'bg-green-100 text-green-800 border border-green-200' : 
            'bg-blue-100 text-blue-800 border border-blue-200'
          }`}>
            {status.type === 'error' ? <AlertTriangle className="w-5 h-5"/> : <CheckCircle className="w-5 h-5"/>}
            <span>{status.message}</span>
          </div>
        )}

        {/* Le formulaire enveloppe maintenant les deux étapes */}
        <form onSubmit={handlePredict} className="p-8 space-y-12">
          
          <div className="grid md:grid-cols-2 gap-12">
            <div className="space-y-6">
              <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">Étape 1</span>
                Références (Support Set)
              </h2>
              <p className="text-sm text-gray-500">Nommez les fichiers avec le prénom de la personne (ex: <b>Lounas.jpg</b>).</p>
              
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center hover:bg-gray-50 transition-colors">
                <input 
                  type="file" multiple accept="image/*"
                  onChange={(e) => setReferenceFiles(Array.from(e.target.files))}
                  className="hidden" id="support-upload"
                />
                <label htmlFor="support-upload" className="cursor-pointer flex flex-col items-center">
                  <Upload className="w-8 h-8 text-blue-500 mb-2" />
                  <span className="text-gray-700 font-medium">Sélectionner des références</span>
                  <span className="text-xs text-gray-400 mt-1">{referenceFiles.length} fichier(s) prêt(s)</span>
                </label>
              </div>
            </div>

            <div className="space-y-6">
              <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">Étape 2</span>
                Photo de Groupe
              </h2>
              <p className="text-sm text-gray-500">L'IA détectera les visages et les comparera aux références fournies.</p>
              
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center hover:bg-gray-50 transition-colors">
                <input 
                  type="file" accept="image/*"
                  onChange={(e) => setGroupFile(e.target.files[0])}
                  className="hidden" id="group-upload"
                />
                <label htmlFor="group-upload" className="cursor-pointer flex flex-col items-center">
                  <ImageIcon className="w-8 h-8 text-blue-500 mb-2" />
                  <span className="text-gray-700 font-medium">Sélectionner la photo globale</span>
                  <span className="text-xs text-gray-400 mt-1">{groupFile ? groupFile.name : 'Aucun fichier'}</span>
                </label>
              </div>
            </div>
          </div>

          <button disabled={loading} className="w-full py-4 bg-blue-900 hover:bg-blue-950 text-white rounded-xl font-bold text-lg shadow-lg transition-all">
            {loading ? 'Traitement en cours...' : 'Lancer la Reconnaissance'}
          </button>
        </form>

        {resultImage && (
          <div className="p-8 border-t border-gray-100 bg-gray-50">
            <h3 className="text-xl font-bold text-gray-800 mb-4 text-center">Résultat de la Détection</h3>
            <div className="rounded-xl overflow-hidden shadow-lg border-4 border-white mx-auto max-w-3xl">
              <img src={resultImage} alt="Résultat" className="w-full h-auto" />
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default DemoReconnaissance;
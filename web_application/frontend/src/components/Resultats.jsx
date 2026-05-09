import { Circle, TrendingUp, Award, Target, BarChart3, CheckCircle2 } from 'lucide-react';
import React from 'react';

const Resultats = () => {
  // Données de performance
  const metricsData = [
    {
      title: "5-way 1-shot",
      accuracy: 75.2,
      precision: 78.5,
      recall: 72.8,
      color: "bg-blue-500"
    },
    {
      title: "10-way 1-shot",
      accuracy: 68.5,
      precision: 71.2,
      recall: 66.3,
      color: "bg-blue-600"
    },
    {
      title: "20-way 1-shot",
      accuracy: 62.3,
      precision: 65.8,
      recall: 59.7,
      color: "bg-blue-700"
    }
  ];

  const comparisons = [
    { model: "Matching Networks", accuracy: 43.6, year: "2016" },
    { model: "MAML", accuracy: 48.7, year: "2017" },
    { model: "Relation Network", accuracy: 50.4, year: "2018" },
    { model: "ProtoNet (Baseline)", accuracy: 49.4, year: "2017" },
    { model: "Notre Modèle (ProtoNet + ResNet-18)", accuracy: 75.2, year: "2024", highlight: true }
  ];

  const highlights = [
    {
      icon: <Award className="w-8 h-8" />,
      title: "+25% vs État de l'Art",
      description: "Amélioration significative sur 5-way 1-shot",
      color: "text-green-600",
      bgColor: "bg-green-50"
    },
    {
      icon: <Target className="w-8 h-8" />,
      title: "62.3% en 20-way",
      description: "Performance robuste sur scénarios complexes",
      color: "text-blue-600",
      bgColor: "bg-blue-50"
    },
    {
      icon: <TrendingUp className="w-8 h-8" />,
      title: "Corrélation r=0.78",
      description: "Qualité d'image ↔ Reconnaissance validée",
      color: "text-purple-600",
      bgColor: "bg-purple-50"
    }
  ];

  return (
    <section 
      id='Resultats' 
      className='relative overflow-hidden bg-white py-12 px-4 sm:py-16 md:py-20 md:px-12 lg:px-20'
    >
      <div className='max-w-7xl mx-auto'>
        
        {/* Header */}
        <div className='text-center mb-12' data-aos='fade-down'>
          <h2 className='text-3xl sm:text-4xl md:text-5xl text-gray-900'>
            Résultats et{" "}
            <span className='font-bold text-black'>
              Performances
            </span>
          </h2>
          <div className='flex gap-3 mt-4 justify-center'>
            <Circle className='text-blue-500 w-5 h-5' />
            <Circle className='text-blue-700 w-5 h-5' />
            <Circle className='text-blue-900 w-5 h-5' />
          </div>
          <p className='mt-6 text-lg text-gray-600 max-w-3xl mx-auto'>
            Évaluation rigoureuse sur le dataset WebFace avec protocole episodic training standard
          </p>
        </div>

        {/* Highlights Cards */}
        <div className='grid grid-cols-1 md:grid-cols-3 gap-6 mb-16' data-aos='fade-up'>
          {highlights.map((item, idx) => (
            <div 
              key={idx}
              className={`${item.bgColor} rounded-2xl p-6 border border-gray-100 
              shadow-md hover:shadow-lg transition-all`}
              data-aos='zoom-in'
              data-aos-delay={idx * 100}
            >
              <div className={`${item.color} mb-4`}>
                {item.icon}
              </div>
              <h3 className='text-xl font-bold text-gray-900 mb-2'>
                {item.title}
              </h3>
              <p className='text-gray-600 text-sm'>
                {item.description}
              </p>
            </div>
          ))}
        </div>

        {/* Performance Metrics */}
        <div className='mb-16' data-aos='fade-up' data-aos-delay='200'>
          <h3 className='text-2xl font-bold text-gray-900 mb-8 text-center'>
            Métriques de Performance
          </h3>
          <div className='grid grid-cols-1 md:grid-cols-3 gap-6'>
            {metricsData.map((metric, idx) => (
              <div 
                key={idx}
                className='bg-gradient-to-br from-gray-50 to-blue-50 rounded-2xl p-6 
                border border-gray-100 shadow-md hover:shadow-xl transition-all'
                data-aos='fade-up'
                data-aos-delay={idx * 100}
              >
                <h4 className='text-lg font-semibold text-gray-800 mb-4'>
                  {metric.title}
                </h4>
                
                {/* Accuracy */}
                <div className='mb-4'>
                  <div className='flex justify-between text-sm mb-2'>
                    <span className='text-gray-600'>Accuracy</span>
                    <span className='font-bold text-blue-600'>{metric.accuracy}%</span>
                  </div>
                  <div className='w-full bg-gray-200 rounded-full h-3'>
                    <div 
                      className={`${metric.color} h-3 rounded-full transition-all duration-1000`}
                      style={{ width: `${metric.accuracy}%` }}
                    />
                  </div>
                </div>

                {/* Precision */}
                <div className='mb-4'>
                  <div className='flex justify-between text-sm mb-2'>
                    <span className='text-gray-600'>Précision</span>
                    <span className='font-bold text-blue-600'>{metric.precision}%</span>
                  </div>
                  <div className='w-full bg-gray-200 rounded-full h-3'>
                    <div 
                      className={`${metric.color} h-3 rounded-full transition-all duration-1000`}
                      style={{ width: `${metric.precision}%` }}
                    />
                  </div>
                </div>

                {/* Recall */}
                <div>
                  <div className='flex justify-between text-sm mb-2'>
                    <span className='text-gray-600'>Rappel</span>
                    <span className='font-bold text-blue-600'>{metric.recall}%</span>
                  </div>
                  <div className='w-full bg-gray-200 rounded-full h-3'>
                    <div 
                      className={`${metric.color} h-3 rounded-full transition-all duration-1000`}
                      style={{ width: `${metric.recall}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Comparison Table */}
        <div className='mb-16' data-aos='fade-up' data-aos-delay='300'>
          <h3 className='text-2xl font-bold text-gray-900 mb-8 text-center'>
            Comparaison avec l'État de l'Art
          </h3>
          <div className='bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100'>
            <div className='overflow-x-auto'>
              <table className='w-full'>
                <thead className='bg-gradient-to-r from-blue-600 to-blue-800 text-white'>
                  <tr>
                    <th className='px-6 py-4 text-left text-sm font-semibold'>Modèle</th>
                    <th className='px-6 py-4 text-center text-sm font-semibold'>Accuracy (5-way 1-shot)</th>
                    <th className='px-6 py-4 text-center text-sm font-semibold'>Année</th>
                    <th className='px-6 py-4 text-center text-sm font-semibold'>Status</th>
                  </tr>
                </thead>
                <tbody className='divide-y divide-gray-100'>
                  {comparisons.map((model, idx) => (
                    <tr 
                      key={idx}
                      className={`${model.highlight ? 'bg-green-50 font-semibold' : 'hover:bg-gray-50'} 
                      transition-colors`}
                    >
                      <td className='px-6 py-4 text-sm text-gray-900'>
                        {model.model}
                      </td>
                      <td className='px-6 py-4 text-center'>
                        <span className={`text-lg font-bold ${model.highlight ? 'text-green-600' : 'text-gray-700'}`}>
                          {model.accuracy}%
                        </span>
                      </td>
                      <td className='px-6 py-4 text-center text-sm text-gray-600'>
                        {model.year}
                      </td>
                      <td className='px-6 py-4 text-center'>
                        {model.highlight ? (
                          <span className='inline-flex items-center gap-1 px-3 py-1 bg-green-100 
                          text-green-700 rounded-full text-xs font-medium'>
                            <CheckCircle2 className='w-4 h-4' />
                            Notre Modèle
                          </span>
                        ) : (
                          <span className='text-gray-400 text-xs'>Baseline</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Calibration Results */}
        <div className='bg-gradient-to-br from-purple-50 to-blue-50 rounded-3xl p-8 md:p-12 border border-purple-100' 
          data-aos='fade-up' 
          data-aos-delay='400'
        >
          <div className='max-w-4xl mx-auto text-center'>
            <BarChart3 className='w-12 h-12 text-purple-600 mx-auto mb-6' />
            <h3 className='text-2xl md:text-3xl font-bold text-gray-900 mb-4'>
              Validation de la Métrique de Qualité
            </h3>
            <p className='text-gray-700 mb-8 text-lg'>
              Notre protocole de calibration a démontré une forte corrélation (r = 0.78, p inférieure à 0.01) 
              entre le score de qualité d'image et les performances de reconnaissance.
            </p>
            <div className='grid grid-cols-1 md:grid-cols-3 gap-6 text-left'>
              <div className='bg-white rounded-xl p-6 shadow-md'>
                <div className='text-3xl font-bold text-purple-600 mb-2'>0.78</div>
                <div className='text-sm text-gray-600'>Corrélation de Pearson</div>
              </div>
              <div className='bg-white rounded-xl p-6 shadow-md'>
                <div className='text-3xl font-bold text-purple-600 mb-2'>p inférieure à 0.01</div>
                <div className='text-sm text-gray-600'>Significativité statistique</div>
              </div>
              <div className='bg-white rounded-xl p-6 shadow-md'>
                <div className='text-3xl font-bold text-purple-600 mb-2'>85%</div>
                <div className='text-sm text-gray-600'>Images validées (score supérieur à 70)</div>
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* Decorative circles */}
      <div className='hidden md:block absolute border-2 border-blue-500 top-20 right-10 
      w-24 h-24 rounded-full opacity-20' data-aos='zoom-in' data-aos-delay='600' />
      <div className='hidden md:block absolute border-2 border-purple-500 bottom-20 left-10 
      w-32 h-32 rounded-full opacity-20' data-aos='zoom-in' data-aos-delay='700' />
    </section>
  );
};

export default Resultats;
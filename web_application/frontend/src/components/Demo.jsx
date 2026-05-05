import { Brush, CogIcon, Check, ArrowBigRight, Circle } from 'lucide-react';
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import howtouseai from '../assets/howtouseai.png';
import avenir from '../assets/a_venir.jpg';

const Demo = () => {
    const [ActiveDemo, setActiveDemo] = useState(1); 

    const demo = [
        {
            id: 1,
            icon: <Brush className='w-5 h-5 sm:w-6 sm:h-6'/>,
            title: "Reconnaissance",
            description: "Testez notre modèle ProtoNet. Enregistrez des visages de référence (enrôlement sous contrôle qualité), puis identifiez-les dans une photo de groupe.",
            features: ["Enrôlement Few-Shot", "Recherche sur image globale", "Similarité Cosinus", "YOLOv8 + ResNet18"],
            color: 'bg-blue-500',
            butColor: 'bg-blue-500 hover:bg-blue-950', 
            iconColor: 'text-blue-500',
            image: howtouseai,
            link: '/demo/reconnaissance'
        },
        {
            id: 2,
            icon: <CogIcon className='w-5 h-5 sm:w-6 sm:h-6'/>,
            title: " Qualité d'image",
            description: "Testez notre vigile hybride. Évaluez la netteté, l'exposition et la géométrie faciale de vos images avant de les intégrer à la base de données.",
            features: ["Filtre Flou (Laplacien)", "Filtre Luminosité", "Score Géométrique (GraFIQs)", "Rejet automatique"],
            color: 'bg-blue-700',
            butColor: 'bg-blue-700 hover:bg-blue-950', 
            iconColor: 'text-blue-700',
            image: avenir,
            link: '/demo/qualite'
        }
    ];

  return (
    <section id='Demo' className='relative overflow-hidden bg-gradient-to-br from-gray-50 to-blue-50 py-12 px-4 sm:py-16 md:px-12 lg:px-20'>
        <div className='max-w-7xl mx-auto'>
            <div className='flex flex-col lg:flex-row items-center justify-center text-center mb-6' data-aos='fade-down'>
                <div className='flex-1 max-w-2xl mx-auto space-y-6 mb-10 lg:mb-0'>
                    <div>
                        <h2 className='text-3xl sm:text-4xl md:text-5xl text-gray-900'>
                            Démonstration {" "}
                            <span className='font-bold text-black'>
                                du modèle<span className='text-blue-900'> !</span>
                            </span>
                        </h2>
                        <div className='flex gap-3 mt-4 justify-center '>
                            <Circle className='text-blue-500 w-5 h-5'></Circle>
                            <Circle className='text-blue-700 w-5 h-5'></Circle>
                            <Circle className='text-blue-900 w-5 h-5'></Circle>
                        </div>
                    </div>
                </div>
            </div>

            <div className='flex flex-wrap gap-3 sm:gap-4 mb-8 sm:mb-12 justify-center' data-aos='fade-up' data-aos-delay='100'>
                {demo.map((demoItem) => (
                    <button
                        key={demoItem.id}
                        onClick={() => setActiveDemo(demoItem.id)}
                        className={`px-4 py-2 sm:px-6 sm:py-3 rounded-full font-medium flex items-center gap-2 transition-all text-sm sm:text-base ${
                            ActiveDemo === demoItem.id
                            ? `${demoItem.color} text-white shadow-lg`
                            : 'bg-white text-gray-700 shadow-md hover:shadow-lg'
                        }`}
                        data-aos='fade-up'
                        data-aos-delay={demoItem.id * 100}
                    >
                        {demoItem.icon}
                        {demoItem.title}
                    </button>
                ))}
            </div>

            <div className='bg-white rounded-2xl md:rounded-3xl shadow-lg md:shadow-xl p-6 sm:p-8 mb-12 md:mb-16 border border-gray-100' data-aos='fade-up' data-aos-delay='200'>
                {demo.filter(demoItem => demoItem.id === ActiveDemo).map(demoItem => (
                    <div key={demoItem.id} className='flex flex-col lg:flex-row gap-6 md:gap-10'>
                        
                        <div className='flex-1'>
                            <div className='flex items-center gap-3 sm:gap-4 mb-4 sm:mb-6' data-aos='fade-right' data-aos-delay='300'>
                                <div className={`w-12 h-12 sm:w-16 sm:h-16 rounded-lg sm:rounded-xl flex items-center justify-center ${demoItem.color} bg-opacity-10`}>
                                    <div className={demoItem.iconColor}>
                                        {demoItem.icon}
                                    </div>
                                </div>
                                <h3 className='text-2xl sm:text-3xl font-bold text-gray-800'>
                                    {demoItem.title}
                                </h3>
                            </div>
                            
                            <p className='text-base sm:text-lg text-gray-700 mb-4 sm:mb-6 leading-relaxed' data-aos='fade-right' data-aos-delay='350'>
                                {demoItem.description}
                            </p>
                            
                            <div className='grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 mb-6 sm:mb-8' data-aos='fade-up' data-aos-delay='400'>
                                {demoItem.features.map((feature, index) => (
                                    <div 
                                        key={index}
                                        className='flex items-center gap-2'
                                        data-aos='fade-up'
                                        data-aos-delay={450 + (index * 50)}
                                    >
                                        <Check className='w-4 h-4 sm:w-5 sm:h-5 text-blue-400 flex-shrink-0'/>
                                        <span className='text-gray-700 text-sm sm:text-base'>
                                            {feature}
                                        </span>
                                    </div>
                                ))}
                            </div>

                            <div className='flex justify-center lg:justify-start' data-aos='fade-up' data-aos-delay='600'>
                                <Link 
                                    to={demoItem.link}
                                    className={`px-6 py-3 sm:px-8 ${demoItem.butColor} font-medium transition-all shadow-md text-white rounded-full hover:shadow-lg flex items-center gap-2 text-sm sm:text-base`}
                                >
                                    Essayer
                                    <ArrowBigRight className='w-4 h-4 sm:w-5 sm:h-5' />
                                </Link>
                            </div>
                        </div>

                        <div className='flex-1 mt-6 lg:mt-0' data-aos='zoom-in' data-aos-delay='500'>
                            <div className='w-full h-60 sm:h-72 md:h-80 overflow-hidden shadow-lg rounded-xl md:rounded-2xl'>
                                <img 
                                    src={demoItem.image} 
                                    alt={demoItem.title}
                                    className='object-cover w-full h-full transform hover:scale-105 transition-transform duration-700'
                                />
                            </div>
                        </div>

                    </div>
                ))}
            </div>
        </div>

        {/* Cercles décoratifs en arrière-plan */}
        <div className='hidden md:block absolute border-2 border-blue-500 bottom-20 left-10 w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 rounded-full opacity-50' data-aos='zoom-in' data-aos-delay='700'></div>
        <div className='hidden md:block absolute border-2 border-blue-800 top-40 right-10 w-20 h-20 sm:w-24 sm:h-24 md:w-32 md:h-32 rounded-full opacity-50' data-aos='zoom-in' data-aos-delay='700'></div>

    </section>
  )
}

export default Demo;
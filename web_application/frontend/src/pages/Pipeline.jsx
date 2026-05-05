import React from 'react'
import { Link } from "react-router-dom";
import { ArrowBigLeft, ArrowBigRight,Circle } from 'lucide-react';




const pipeline = () => {
  return (
    <section
    id="Pipeline"
    className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 mt-12
    flex flex-col justify-center items-center text-center p-8 transition-all 
    duration-500 ease-in-out"
    data-aos="zoom-in"
    data-aos-delay="200"
    >
        <div className='mb-6 md:mb-8 ' data-aos="fade-left">
            <h1 className="text-4xl md:text-5xl font-semibold text-gray-900 mb-6">
                Pipeline de <span className='text-blue-900'>Recherche</span>
            </h1>
            <div className='flex gap-3 mt-4 justify-center '>
               <Circle className='text-blue-500 w-5 h-5'></Circle>
               <Circle className='text-blue-700 w-5 h-5'></Circle>
               <Circle className='text-blue-900 w-5 h-5'></Circle>
            </div>
        </div>


        

      <p className="max-w-3xl text-lg md:text-xl text-gray-700 leading-relaxed mb-10">
        Notre pipeline de recherche avancée combine les dernières innovations
        en apprentissage profond et en reconnaissance visuelle. Chaque module
        contribue à l’efficacité globale du modèle :
      </p>
        <ul className="max-w-2xl text-left text-gray-600 space-y-4">
        <li>
          <span className="font-semibold text-blue-800">YOLOv5 :</span> Détection
          rapide et précise des objets sur l’image.
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. 
          Molestiae esse similique quasi laudantium vero maiores,
           qui voluptatum adipisci est culpa reiciendis, 
          vitae ad quisquam repudiandae atque officia eum minima enim.
          rapide et précise des objets sur l’image.
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. 
          Molestiae esse similique quasi laudantium vero maiores,
           qui voluptatum adipisci est culpa reiciendis, 
          vitae ad quisquam repudiandae atque officia eum minima enim.
        </li>
        <li>
          <span className="font-semibold text-blue-800">M.A.L.M :</span> Ajustement
          métrique pour améliorer la compréhension de similarité.
          rapide et précise des objets sur l’image.
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. 
          Molestiae esse similique quasi laudantium vero maiores,
           qui voluptatum adipisci est culpa reiciendis, 
          vitae ad quisquam repudiandae atque officia eum minima enim.
          rapide et précise des objets sur l’image.
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. 
          Molestiae esse similique quasi laudantium vero maiores,
           qui voluptatum adipisci est culpa reiciendis, 
          vitae ad quisquam repudiandae atque officia eum minima enim.
        </li>
        <li>
          <span className="font-semibold text-blue-800">REPTILE :</span> Technique
          d’apprentissage méta pour une adaptation rapide aux nouvelles classes.
          rapide et précise des objets sur l’image.
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. 
          Molestiae esse similique quasi laudantium vero maiores,
           qui voluptatum adipisci est culpa reiciendis, 
          vitae ad quisquam repudiandae atque officia eum minima enim.
          rapide et précise des objets sur l’image.
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. 
          Molestiae esse similique quasi laudantium vero maiores,
           qui voluptatum adipisci est culpa reiciendis, 
          vitae ad quisquam repudiandae atque officia eum minima enim.
          rapide et précise des objets sur l’image.
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. 
          Molestiae esse similique quasi laudantium vero maiores,
           qui voluptatum adipisci est culpa reiciendis, 
          vitae ad quisquam repudiandae atque officia eum minima enim.
        </li>
        <li>
          <span className="font-semibold text-blue-800">SIFT :</span> Extraction de
          caractéristiques locales robustes et invariantes.
          rapide et précise des objets sur l’image.
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. 
          Molestiae esse similique quasi laudantium vero maiores,
           qui voluptatum adipisci est culpa reiciendis, 
          vitae ad quisquam repudiandae atque officia eum minima enim.
          rapide et précise des objets sur l’image.
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. 
          Molestiae esse similique quasi laudantium vero maiores,
           qui voluptatum adipisci est culpa reiciendis, 
          vitae ad quisquam repudiandae atque officia eum minima enim.
          rapide et précise des objets sur l’image.
          Lorem ipsum dolor sit, amet consectetur adipisicing elit. 
          Molestiae esse similique quasi laudantium vero maiores,
           qui voluptatum adipisci est culpa reiciendis, 
          vitae ad quisquam repudiandae atque officia eum minima enim.
        </li>
      </ul>

      <p className="max-w-3xl text-gray-600 mt-12">
        En combinant ces approches, notre système atteint un équilibre entre
        rapidité, précision et généralisation — idéal pour les scénarios de
        reconnaissance à un seul échantillon .
      </p>




      <div className='flex justify-center lg:justify-start mt-8 
      md:mt-10 z-10'
      
      
      >
        <Link 
        to='/'
        className='px-6 py-3 md:px-8 md:py-3 bg-blue-500 
        text-white rounded-full font-medium hover:bg-blue-700
        transition-all shadow-md hover:shadow-lg flex items-center
        gap-2 text-sm md:text-base'>
          Accueil
          <ArrowBigLeft className='w-5 h-5 md:h-5 md:w-5'/>

        </Link>

      </div>

    </section>
  )
}

export default pipeline
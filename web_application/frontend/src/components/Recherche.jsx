import React from 'react'
import { Link } from "react-router-dom"
import recherche from '../assets/recherche_2.jpg'
import { Users,Circle, Bot, Search, Target, ArrowBigRight, User } from 'lucide-react'

const Recherche = () => {
  return (
    <section id='Recherche' 
    className='relative overflow-hidden bg-gradient-to-br
    from-gray-50 to-blue-50 py-12 px-4 sm:py-16 md:py-20 md:px-12
    lg:px-20 flex flex-col lg:flex-row items-center justify-between'>
      <div className='flex-1 w-full lg:mr-8 xl:mr-12 relative order-2 
      lg:order-1 mt-10 lg:mt-0'>
        <div 
        className='w-full maw-w-md mx-auto lg:max-w-lg xl:max-w-xl 
        h-[300px] sm:h-[350px] md:h-[400px] lg:h-[450px] xl:h-[500px]
        overflow-hidden shadow-lg md:shadow-xl relative z-10 -rotate-2
        rounded-[40%_60%_70%_30%/40%_50%_60%_60%] hover:rotate-0
        transition-transform duration-700'
        data-aos="fade-right"
        data-aos-delay="200"
        >
          <img src={recherche} alt="Recherche" 
          className='object-cover w-full h-full transform hover:scale-110 
          transition-transform duration-700 '/>
        </div>
      </div>

      <div className='hidden md:block absolute border-2 border-blue-500 
      bottom-4 right-4 w-16 h-16 md:w-20 md:h-20 lg:w-24 rounded-full
      z-0'
      data-aos='zoom-in'
      data-aos-delay='500'>

      </div>
      <div className='hidden md:block absolute border-2 border-blue-700 
      top-4 left-4 w-16 h-16 md:w-20 md:h-20 lg:w-24 rounded-full
      z-0'
      data-aos='zoom-in'
      data-aos-delay='600'>        
      </div>

      <div className='flex-1 w-full max-w-2xl mx-auto lg:mx-0 space-y-6
      md:space-y-8 relative z-20 order-1 lg:order-2' >
        <div className='mb-6 md:mb-8 ' data-aos="fade-left">
          <h2 className='text-3xl sm:text-4xl md:text-5xl 
          text-gray-900 text-center lg:text-left'>
            Pipeline {""}
            <span className='font-bold text-black block lg:inline'>
              de recherche 
              <span className='text-blue-500'> avancée</span>
            </span>

          </h2>
          <div className='flex gap-3 mt-4 justify-center lg:justify-start'>
            <Circle className='text-blue-500 w-5 h-5'></Circle>
            <Circle className='text-blue-700 w-5 h-5'></Circle>
            <Circle className='text-blue-900 w-5 h-5'></Circle>
          </div>
        </div>
        <p className='text-base sm:text-lg text-gray-700 leading-relaxed
        text-center lg:text-left'
        data-aos="fade-left"
        data-aos-delay="100">
          Lorem ipsum dolor sit amet consectetur adipisicing elit. 
          Doloremque rerum aut laborum. Perspiciatis, quos. Illo,
          numquam harum doloremque eveniet reprehenderit nemo excepturi impedit
          alias. Alias laudantium tempore natus eveniet quo?

        </p>
        <div className='grid grid-cols-1 sm:grid-cols-2 gap-4 
        md:gap-6 mt-8 md:mt-10'
        data-aos="fade-up"
        data-aos-delay="200"
        >
          <div className='bg-white p-4 md:p-6 rounded-xl md:rounded-2xl 
          border border-gray-100 shadow-md md:shadow-lg transition-all
          hover:shadow-xl'>
            <div className='w-10 h-10 md:w-12 md:h-12 flex items-center 
            justify-center rounded-lg bg-blue-100 mb-3 md:mb-4 '>
              <Bot className='text-blue-500 w-5 h-5 md:w-6 md:h-6 '/>
            </div>
            <h3 className='text-base md:text-lg font-semibold text-gray-800
            mb-2'> YOLOv5
            </h3>
            <p className='text-gray-600 text-xs md:text-sm'> 
              Lorem ipsum dolor sit amet consectetur adipisicing elit.
            </p> 
          </div>

          <div className='bg-white p-4 md:p-6 rounded-xl md:rounded-2xl 
          border border-gray-100 shadow-md md:shadow-lg transition-all
          hover:shadow-xl'>
            <div className='w-10 h-10 md:w-12 md:h-12 flex items-center 
            justify-center rounded-lg bg-blue-100 mb-3 md:mb-4 '>
              <Search className='text-blue-700 w-5 h-5 md:w-6 md:h-6 '/>
            </div>
            <h3 className='text-base md:text-lg font-semibold text-gray-800
            mb-2'> MALM
            </h3>
            <p className='text-gray-600 text-xs md:text-sm'> 
              Lorem ipsum dolor sit amet consectetur adipisicing elit.
            </p> 
          </div>

          <div className='bg-white p-4 md:p-6 rounded-xl md:rounded-2xl 
          border border-gray-100 shadow-md md:shadow-lg transition-all
          hover:shadow-xl'>
            <div className='w-10 h-10 md:w-12 md:h-12 flex items-center 
            justify-center rounded-lg bg-blue-100 mb-3 md:mb-4 '>
              <Search className='text-blue-700 w-5 h-5 md:w-6 md:h-6 '/>
            </div>
            <h3 className='text-base md:text-lg font-semibold text-gray-800
            mb-2'> REPTILE
            </h3>
            <p className='text-gray-600 text-xs md:text-sm'> 
              Lorem ipsum dolor sit amet consectetur adipisicing elit.
            </p> 
          </div>

          <div className='bg-white p-4 md:p-6 rounded-xl md:rounded-2xl 
          border border-gray-100 shadow-md md:shadow-lg transition-all
          hover:shadow-xl'>
            <div className='w-10 h-10 md:w-12 md:h-12 flex items-center 
            justify-center rounded-lg bg-blue-100 mb-3 md:mb-4 '>
              <Users className='text-blue-900 w-5 h-5 md:w-6 md:h-6 '/>
            </div>
            <h3 className='text-base md:text-lg font-semibold text-gray-800
            mb-2'> SIFT
            </h3>
            <p className='text-gray-600 text-xs md:text-sm'> 
              Lorem ipsum dolor sit amet consectetur adipisicing elit.
            </p> 
          </div>


        </div>
      <div className='flex justify-center lg:justify-start mt-8 
      md:mt-10 '
      data-aos="fade-up"
      data-aos-delay="300"
      >
        <Link 
        to='/Pipeline'
        className='px-6 py-3 md:px-8 md:py-3 bg-blue-500 
        text-white rounded-full font-medium hover:bg-blue-700
        transition-all shadow-md hover:shadow-lg flex items-center
        gap-2 text-sm md:text-base'>
          En savoir plus
          <ArrowBigRight className='w-5 h-5 md:h-5 md:w-5'/>

        </Link>

      </div>

      </div>

    </section>
  )
}

export default Recherche
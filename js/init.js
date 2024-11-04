/*
(function($){
  $(function(){
    AOS.init();
    
    $('.scrollspy').scrollSpy();
    $('.parallax').parallax();
    $(".button-collapse").sideNav();
  }); // end of document ready
})(jQuery); // end of jQuery name space
*/
$( document ).ready(function(){

	AOS.init();
    
    $('.scrollspy').scrollSpy();
    $('.parallax').parallax();
    $(".button-collapse").sideNav();

  $(window).scroll(function() {
    if ($(document).scrollTop() > 250) {
      $('nav').addClass('white');
    } else {
      $('nav').removeClass('white');
    }
  });

})
c-----------------------------------------------------------------------
      subroutine uservp (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      udiff =0.
      utrans=0.

      return
      end
c-----------------------------------------------------------------------
      subroutine userf  (ix,iy,iz,ieg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      ffx = 0.0 
      ffy = 0.0 
      ffz = 0.0

      return
      end
c-----------------------------------------------------------------------
      subroutine userq  (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      qvol   = 0.0
      source = 0.0

      return
      end
c-----------------------------------------------------------------------
      subroutine userchk
      include 'SIZE'
      include 'TOTAL'

      if (istep.eq.0) time=0

c     Only write output every 100th step after frist 100
      nio = -1
      if (istep.le.100.or.mod(istep,100).eq.0) nio=nid

      call write_stuff

      return
      end

c-----------------------------------------------------------------------
      subroutine userbc(ix,iy,iz,iside,eg) ! set up boundary conditions
c
c     NOTE ::: This subroutine MAY NOT be called by every process
c
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      integer e, bID

      e = gllel(ieg)
      bID = boundaryID(iside,e)
      pa = 0.0

      if (cbu.eq.'o  ') then
         U0 = 1.0                  ! characteristic velocity
         delta = 0.1               ! small positive constant
         pa = dongOutflow(ix,iy,iz,e,iside,U0,delta)
      else 
         ux = 1.0
         uy = 0.0
         uz = 0.0
      endif 

      return
      end
c-----------------------------------------------------------------------
      subroutine useric (ix,iy,iz,ieg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'

      ux=1.0
      uy=0.0
      uz=0.0
      temp=0

      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat
      include 'SIZE'
      include 'TOTAL'

      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat2
      include 'SIZE'
      include 'TOTAL'

      ntot = nx1*ny1*nz1*nelt

c     Set boundary conditions for gmsh lines
c     To use outflow from Dong et al, set outlet BC to 'o  '
      do iel=1,nelv
      do ifc=1,2*ndim
         id_face = bc(5,ifc,iel,1)
         if (id_face.eq.1) then        ! surface 1 for inlet 
            cbc(ifc,iel,1) = 'v  '
         elseif (id_face.eq.2) then    ! surface 2 for outlet
            cbc(ifc,iel,1) = 'o  '
         elseif (id_face.eq.3) then    ! surface 3 for sym
            cbc(ifc,iel,1) = 'SYM'
         elseif (id_face.eq.4) then    ! surface 4 for wall
            cbc(ifc,iel,1) = 'W  '
        endif
      enddo
      enddo
      
      return
      end

c-----------------------------------------------------------------------
      subroutine usrdat3
      include 'SIZE'
      include 'TOTAL'
c
      return
      end
c----------------------------------------------------------------------

      subroutine write_stuff
      include 'SIZE'
      include 'TOTAL'

      real x0(3)
      save x0
      data x0 /3*0/

      integer bIDs(1), iobj_wall(1)
      parameter (lt=lx1*ly1*lz1*lelv)

      character(len=60) :: fmt  ! Format for writing lift coefficients
      fmt = "(f9.3, TR4, f9.6, TR4, f9.6, TR4, f9.6:)"

c     define objects for surface integrals (for aerodynamic coefficients)
      if (istep.eq.0) then
         bIDs(1) = 4   ! Line 4 is the airfoil
         call create_obj(iobj_wall(1),bIDs,1)
         open(12, file = 'forceCoeffs.dat', access = 'append')  
      endif 

c     Compute lift/drag coefficients on the cylinder
      scale = 2.0  ! Cd = F/(.5 rho U^2)
      call torque_calc(scale,x0,.true.,.true.)
      write(12,fmt) time, dragx(1), dragy(1), torqz(1)

      return
      end

c-----------------------------------------------------------------------

      function dongOutflow(ix,iy,iz,iel,iside,u0,delta)

      include 'SIZE'
      include 'SOLN'
      include 'GEOM'

      real sn(3)

      ux = vx(ix,iy,iz,iel)
      uy = vy(ix,iy,iz,iel)
      uz = vz(ix,iy,iz,iel)

      call getSnormal(sn,ix,iy,iz,iside,iel)
      vn = ux*sn(1) + uy*sn(2) + uz*sn(3) 
      S0 = 0.5*(1.0 - tanh(vn/u0/delta))

      dongOutflow = -0.5*(ux*ux+uy*uy+uz*uz)*S0

      return
      end

c-----------------------------------------------------------------------

c automatically added by makenek
      subroutine usrdat0() 

      return
      end

c automatically added by makenek
      subroutine usrsetvert(glo_num,nel,nx,ny,nz) ! to modify glo_num
      integer*8 glo_num(1)

      return
      end

c automatically added by makenek
      subroutine userqtl

      call userqtl_scig

      return
      end

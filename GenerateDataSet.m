function [data,intrinsic] = GenerateDataSet(N,exampleName,noiselevel)
%GENERATEDATASET with N data points on surface =
%'sphere','torus','rp2','kleinbottle'. or 'doubletorus'

    if (nargin<3)
        noiselevel=0;
    end
    
    switch lower(exampleName)
        
        case 'circle'
            
            theta = 2*pi*(1/N:1/N:1);

            data = [cos(theta);sin(theta);zeros(size(theta))];
            
            intrinsic = [cos(theta);sin(theta)];
            
        case 'sphere'
            
            X = randn(3,N);
            data = X./repmat(sqrt(sum(X.^2)),3,1);     %%% points on the sphere
            XX=X(1,:)./(1-X(3,:));
            YY=X(2,:)./(1-X(3,:));
            RR = sqrt(XX.^2+YY.^2);
            XX = XX.*log(1+RR)./RR;
            YY = YY.*log(1+RR)./RR;
            intrinsic = [XX;YY];
            
                        
        case 'flattorus'
            
            NN=floor(sqrt(N));
            t=1/NN:1/NN:1;              	%%% a grid of point in [0,1]
            NN=length(t);
            theta=repmat(2*pi*t,1,NN);    
            phi = repmat(2*pi*t,NN,1);
            phi = phi(:)';              %%% (theta,phi) is a grid in [0,2*pi]^2

            intrinsic = [theta;phi];

            data=[cos(theta); sin(theta); cos(phi); sin(phi)];
            
        case 'torus'
            
            NN=floor(sqrt(N));
            t=1/NN:1/NN:1;              	%%% a grid of point in [0,1]
            NN=length(t);
            theta=repmat(2*pi*t,1,NN);    
            phi = repmat(2*pi*t,NN,1);
            phi = phi(:)';              %%% (theta,phi) is a grid in [0,2*pi]^2

            intrinsic = [theta;phi];

            R=2; r=1;                 	%%% embed [0,2*pi]^2 into R^3, standard T^2
            x=(R+r*cos(theta)).*cos(phi);
            y=(R+r*cos(theta)).*sin(phi);
            z=r.*sin(theta);

            data=[x; y; z];
            
        case 'mobius'
            
            NN=floor(sqrt(N));
            t=1/NN:1/NN:1;              	%%% a grid of point in [0,1]
            NN=length(t);
            theta=repmat(2*t-1,1,NN);    
            phi = repmat(2*pi*t,NN,1);
            phi = phi(:)';              %%% (theta,phi) is a grid in [0,2*pi]^2

            intrinsic = [theta;phi];
            
            R=1;r=1;
            x = (R+r*theta.*cos(phi/2)/2).*cos(phi);
            y = (R+r*theta.*cos(phi/2)/2).*sin(phi);
            z = r*theta.*sin(phi/2)/2;
            
            data = [x;y;z];

        case 'rp2'

            X = randn(3,N);
            X = X./repmat(sqrt(sum(X.^2)),3,1);     %%% points on the sphere
            x=X(1,:);y=X(2,:);z=X(3,:);
            % (xy, xz, y2?z2, 2yz)
            data = [x.*y; x.*z; y.^2-z.^2; 2*y.*z];
            %data = [x.*y; x.*z; y.*z; x.^2+2*y.^2+3*z.^2];
            XX=X(1,:)./(1-X(3,:));
            YY=X(2,:)./(1-X(3,:));
            RR = sqrt(XX.^2+YY.^2);
            XX = XX.*log(1+RR)./RR;
            YY = YY.*log(1+RR)./RR;
            intrinsic = [XX;YY];

        case 'kleinbottle'

            NN=floor(sqrt(N));
            t=1/NN:1/NN:1;              	%%% a grid of point in [0,1]
            NN=length(t);
            theta=repmat(2*pi*t,1,NN);    
            phi = repmat(2*pi*t,NN,1);
            phi = phi(:)';              %%% (theta,phi) is a grid in [0,2*pi]^2

            intrinsic = [theta;phi];

            R=2; r=1;                 	%%% embed [0,2*pi]^2 into R^3, standard T^2
            x=(R+r*cos(theta)).*cos(phi);
            y=(R+r*cos(theta)).*sin(phi);
            z=r.*sin(theta).*cos(phi/2);
            zz=r.*sin(theta).*sin(phi/2);

            data=[x; y; z; zz]; 

        case 'doubletorus'
            
            NN=floor(sqrt(N/2));
            t=1/NN:1/NN:1;              	%%% a grid of point in [0,1]
            NN=length(t);
            theta=repmat(2*pi*t,1,NN);    
            phi = repmat(2*pi*t,NN,1);
            phi = phi(:)';              %%% (theta,phi) is a grid in [0,2*pi]^2

            R=2; r=1;                 	%%% embed [0,2*pi]^2 into R^3, standard T^2
            x=(R+r*cos(theta)).*cos(phi);
            y=(R+r*cos(theta)).*sin(phi);
            z=r.*sin(theta);
            
            intrinsic = [theta(x<2)-2*pi theta(x>-2);phi(x<2) phi(x>-2)];

            data=[x(x<2)-2-pi/NN/4,x(x>-2)+2+pi/NN/4;y(x<2),y(x>-2);z(x<2),z(x>-2)]; 

        case 'l63'
            
            dt=0.05;
            [x,~] = L63(randn(3,1),(N+10000)*dt,dt,0);
            x=x(10001:end,:)';
            data = x;
            intrinsic = [x(1,:)+x(2,:);x(3,:)];
            
    end

    data = data + noiselevel*randn(size(data));

end
    

function [x,t] = L63(x0,T,tau,D)

    t = 0:tau:T;
    N = length(t);

    x = zeros(N,size(x0,1),size(x0,2));

    x(1,:,:) = x0;
    state = x0;

    for i = 2:N

        %%% Integrate with RK4 and 10 substeps per discrete time step
        for jj=1:10
            k1=(tau/10)*LorenzODE(state);
            k2=(tau/10)*LorenzODE(state+k1/2);
            k3=(tau/10)*LorenzODE(state+k2/2);
            k4=(tau/10)*LorenzODE(state+k3);
            state=state+k1/6+k2/3+k3/3+k4/6;
            state = state + D*sqrt(2)*sqrt(tau/10)*randn(size(state));
        end
        x(i,:,:)=state;

    end

end


function dx = LorenzODE(x)
    rho = 28; sigma = 10; beta = 8/3;
    dx = zeros(3,size(x,2));
    dx(1,:) = sigma*(x(2,:) - x(1,:));
    dx(2,:) = x(1,:).*(rho - x(3,:)) - x(2,:);
    dx(3,:) = x(1,:).*x(2,:) - beta*x(3,:);
end
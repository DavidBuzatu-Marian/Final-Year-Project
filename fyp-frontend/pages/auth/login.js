import React from 'react';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';

const login = () => {
  return (
    <section>
      <Container
        maxWidth='sm'
        sx={{
          minHeight: '100vh',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Card sx={{ mx: 'auto', my: '0', minWidth: '100%' }}>
          <CardContent>
            <Typography variant='h1' component='div'>
              Login to FYP
            </Typography>
          </CardContent>
          <CardActions>
            <Button size='small'>Learn More</Button>
          </CardActions>
        </Card>
      </Container>
    </section>
  );
};

export default login;
